"""LangGraph retrieval and answer steps for research-paper questions."""

import json
from time import perf_counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .db import new_id, open_database
from .grounding import (
    DraftAnswer,
    abstention,
    draft_answer,
    format_answer,
    source_cards,
    verify_claims,
)
from .llm import generate
from .paper_store import source_rows
from .question_planning import QuestionPlan, plan_question
from .quiz_validation import (
    QuizContent,
    QuizRequest,
    parse_quiz_command,
    public_quiz,
    validate_source_ids,
)
from .retrieval import CONTEXT_CHARACTERS, search


def label_evidence(evidence: list[dict]) -> list[dict]:
    """Give the local model short citation labels it can copy reliably."""

    return [
        {**row, "chunk_id": row["id"], "id": f"S{number}"}
        for number, row in enumerate(evidence, 1)
    ]


def quiz_payload(request: QuizRequest, evidence: list[dict]) -> dict:
    """Make grounded quiz questions and reject invented source references."""

    if not evidence:
        raise ValueError("No indexed paper evidence found for this quiz")
    evidence = label_evidence(evidence)
    ids = ", ".join(row["id"] for row in evidence)
    quiz = generate(
        QuizContent,
        f"Create exactly {request.count} multiple-choice questions about the topic in the request. "
        f"Difficulty: {request.difficulty}. Easy=recall; medium=application; hard=scenarios. "
        f"Use only source IDs {ids}. Four distinct options, zero-based correct_index, "
        "an explanation, and source_ids for each question. Do not invent facts.",
        evidence,
        {"topic": request.topic},
    )
    if len(quiz.questions) != request.count:
        raise ValueError("Wrong question count returned; try fewer questions")
    if (
        len({question.question.strip().casefold() for question in quiz.questions})
        != request.count
    ):
        raise ValueError("Duplicate quiz questions returned")
    for question in quiz.questions:
        validate_source_ids(question.source_ids, evidence)
    return {
        "topic": request.topic,
        "difficulty": request.difficulty,
        **quiz.model_dump(),
        "sources": source_cards(evidence),
    }


def make_quiz(request: QuizRequest) -> dict:
    """Save a quiz while leaving the answer key on the server until submission."""

    payload = quiz_payload(request, search(request.topic))
    quiz_id = new_id()
    with open_database() as database:
        database.execute(
            "INSERT INTO quizzes (id, payload) VALUES (?, ?)",
            (quiz_id, json.dumps(payload)),
        )
    return {"kind": "quiz", "quiz_id": quiz_id, **public_quiz(payload)}


WORKFLOW_VERSION = "reviewed-plan-v2"


class ChatState(TypedDict, total=False):
    """One question's plan, evidence, draft, verification and final response."""

    message: str
    history: list[dict]
    tasks: list[dict]
    clarification: str
    evidence: list[dict]
    draft: DraftAnswer
    verification: dict
    result: dict
    retrieval_trace: dict
    timings: dict
    capture_evidence: bool
    require_review: bool
    approved_plan: dict
    awaiting_approval: bool
    deferred_tasks: list[dict]
    force_review: bool


def plan(state: ChatState) -> dict:
    """Resolve requested papers and create bounded search tasks before retrieval."""
    started = perf_counter()
    papers = source_rows()
    approved = state.get("approved_plan")
    planned = (
        QuestionPlan.model_validate(approved)
        if approved
        else plan_question(state["message"], state.get("history", []), papers)
    )
    if approved and any(
        task.paper and task.paper not in {row["title"] for row in papers}
        for task in planned.tasks
    ):
        planned = QuestionPlan(
            clarification="The indexed papers changed. Please revise the plan before approving it again."
        )
    tasks = [task.model_dump() for task in planned.tasks]
    review = (
        state.get("require_review", False)
        and not approved
        and (
            state.get("force_review", False)
            or len(tasks) > 1
            or planned.route != "direct"
        )
    )
    # One response has a finite evidence window; keep larger plans for subsequent batches.
    batch_size = max(1, CONTEXT_CHARACTERS // 2400)
    deferred = tasks[batch_size:] if not review else []
    if not review:
        tasks = tasks[:batch_size]
    return {
        "tasks": tasks,
        "awaiting_approval": review,
        "deferred_tasks": deferred,
        "clarification": planned.clarification,
        "timings": {"planning": perf_counter() - started},
        "retrieval_trace": {
            "route": planned.route,
            "workflow": WORKFLOW_VERSION,
            "approval_mode": "user_approved"
            if approved
            else "review_required"
            if state.get("require_review")
            else "automatic_offline",
            "planned_task_count": len(planned.tasks),
        },
    }


def after_plan(state: ChatState) -> str:
    """Clarify unresolved requests before spending time searching or answering."""
    return (
        "format_response"
        if state.get("clarification") or state.get("awaiting_approval")
        else "retrieve"
    )


def retrieve(state: ChatState) -> dict:
    """Run the same search for each task, sharing a bounded context across tasks."""
    started = perf_counter()
    tasks = state["tasks"]
    groups, runs = [], []
    for task in tasks:
        trace: dict = {"capture_evidence": state.get("capture_evidence", False)}
        groups.append(search(task["question"], paper=task["paper"], diagnostics=trace))
        runs.append({"task": task, "trace": trace})
    # Round-robin selection gives each task a chance before spending the budget.
    remaining, evidence = CONTEXT_CHARACTERS, {}
    task_budget = CONTEXT_CHARACTERS // len(tasks)
    spent = [0] * len(tasks)
    for position in range(max((len(group) for group in groups), default=0)):
        for task_id, group in enumerate(groups):
            if position >= len(group):
                continue
            row = group[position]
            if row["id"] in evidence:
                evidence[row["id"]]["task_ids"].append(task_id)
                continue
            budget = min(remaining, task_budget - spent[task_id])
            # Keep full windows: truncating from the front could discard the child hit.
            if len(row["content"]) > budget:
                continue
            evidence[row["id"]] = {**row, "task_ids": [task_id]}
            remaining -= len(row["content"])
            spent[task_id] += len(row["content"])
    passages = label_evidence(list(evidence.values()))
    trace = {**state["retrieval_trace"], "tasks": runs, "evidence_count": len(passages)}
    for field in ("candidate_sources", "reranked_sources"):
        trace[field] = sorted(
            {title for run in runs for title in run["trace"].get(field, [])}
        )
    for field in (
        "candidate_count",
        "accepted_count",
        "retrieval_seconds",
        "reranking_seconds",
        "expansion_seconds",
    ):
        trace[field] = sum(run["trace"].get(field, 0) for run in runs)
    return {
        "evidence": passages,
        "retrieval_trace": trace,
        "timings": {**state.get("timings", {}), "retrieval": perf_counter() - started},
    }


def after_retrieval(state: ChatState) -> str:
    """Skip generation entirely when no evidence survived retrieval."""
    return "generate" if state["evidence"] else "format_response"


def generate_answer(state: ChatState) -> dict:
    """Generate a structured draft using the original question and grouped evidence."""
    started = perf_counter()
    history = [
        {"role": item["role"], "content": item["content"][:500]}
        for item in state.get("history", [])[-2:]
    ]
    draft = draft_answer(state["message"], history, state["evidence"], state["tasks"])
    return {
        "draft": draft,
        "timings": {**state["timings"], "generation": perf_counter() - started},
    }


def verify(state: ChatState) -> dict:
    """Check source support and declared coverage without another model call."""
    started = perf_counter()
    checked = verify_claims(state["draft"], state["evidence"], state["tasks"])
    return {
        "verification": checked,
        "timings": {**state["timings"], "verification": perf_counter() - started},
    }


def format_response(state: ChatState) -> dict:
    """Build every graph outcome in one place while retaining the browser contract."""
    if state.get("clarification"):
        result = {
            "kind": "answer",
            "status": "clarification",
            "answer": state["clarification"],
            "sources": [],
            "claims": [],
        }
    elif state.get("awaiting_approval"):
        result = {
            "kind": "answer",
            "status": "awaiting_approval",
            "answer": "Please review this plan. Approve it to start searching, or describe what to revise.",
            "sources": [],
            "claims": [],
        }
        result["plan"] = review_plan(
            state["message"], state["tasks"], state["retrieval_trace"]["route"]
        )
        result["answer"] += "\n\n" + "\n".join(
            f"{i + 1}. {task['question']}"
            + (f" — {task['paper']}" if task.get("paper") else "")
            for i, task in enumerate(state["tasks"])
        )
    elif not state.get("evidence"):
        result = abstention(
            "No matching research-paper evidence was found for this question."
        )
    else:
        result = format_answer(
            state["draft"], state["verification"], state["evidence"], state["tasks"]
        )
    if state.get("deferred_tasks"):
        if not state.get("evidence"):
            result["answer"] = (
                "No matching evidence was found for the tasks in this execution batch."
            )
        result["answer"] += (
            f"\n\nThis response covers one execution batch. {len(state['deferred_tasks'])} search tasks remain; approve the next batch to continue."
        )
        result["status"] = "partial" if result.get("claims") else result["status"]
        result["plan"] = review_plan(
            state["message"], state["deferred_tasks"], state["retrieval_trace"]["route"]
        )
        result["answer"] += "\n\nRemaining tasks:\n" + "\n".join(
            f"{i + 1}. {task['question']}"
            + (f" — {task['paper']}" if task.get("paper") else "")
            for i, task in enumerate(state["deferred_tasks"])
        )
    return {"result": result}


def review_plan(question: str, tasks: list[dict], route: str) -> dict:
    """Give each reviewable proposal a new token tied to its exact saved content."""
    return {"plan_id": new_id(), "question": question, "tasks": tasks, "route": route}


workflow = StateGraph(ChatState)
workflow.add_node("plan", plan)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate_answer)
workflow.add_node("verify", verify)
workflow.add_node("format_response", format_response)
workflow.add_edge(START, "plan")
workflow.add_conditional_edges(
    "plan", after_plan, {name: name for name in ("retrieve", "format_response")}
)
workflow.add_conditional_edges(
    "retrieve",
    after_retrieval,
    {name: name for name in ("generate", "format_response")},
)
workflow.add_edge("generate", "verify")
workflow.add_edge("verify", "format_response")
workflow.add_edge("format_response", END)
chat_graph = workflow.compile()


def answer_with_context(
    message: str,
    history: list[dict] | None = None,
    capture_evidence: bool = True,
    require_review: bool = False,
    approved_plan: dict | None = None,
    force_review: bool = False,
) -> dict:
    """Run the shared browser/evaluation workflow; reference answers never enter it."""
    started = perf_counter()
    state = chat_graph.invoke(
        {
            "message": message,
            "history": history or [],
            "capture_evidence": capture_evidence,
            "require_review": require_review,
            "approved_plan": approved_plan,
            "force_review": force_review,
        }
    )
    return {
        **state["result"],
        "context": state.get("evidence", []),
        "retrieval_trace": state.get("retrieval_trace", {}),
        "timings": {**state.get("timings", {}), "total": perf_counter() - started},
    }


def list_papers() -> dict:
    """Build the paper catalog directly from the index without a model call."""

    papers = source_rows()
    titles = "\n".join(row["title"] for row in papers)
    sources = [
        {
            "id": f"S{index}",
            "title": row["title"],
            "source_url": row["url"],
            "location": "Indexed paper",
        }
        for index, row in enumerate(papers, 1)
    ]
    return {
        "kind": "answer",
        "status": "answered",
        "answer": f"Indexed papers:\n\n{titles}"
        if papers
        else "No papers are indexed yet.",
        "sources": sources,
    }


def respond(message: str, history: list[dict], force_review: bool = False) -> dict:
    """Route explicit commands first, then pass questions to the research graph."""

    if message.strip().casefold() == "/papers":
        return list_papers()
    if message.strip().startswith("/quiz"):
        return make_quiz(parse_quiz_command(message))
    result = answer_with_context(
        message,
        history,
        capture_evidence=False,
        require_review=True,
        force_review=force_review,
    )
    result.pop("context")
    return result
