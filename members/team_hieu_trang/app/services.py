"""Coordinate the question-to-answer workflow using LangGraph.

The graph plans search tasks, retrieves evidence, asks Gemma for a draft, verifies
its references and formats a response. Conditional edges stop for clarification,
plan approval or missing evidence. Chat commands and quizzes enter through
handle_chat_message().
"""

import json
from time import perf_counter
from typing import TypedDict
from typing import NotRequired

from langgraph.graph import END, START, StateGraph

from .db import create_record_id, open_database
from .grounding import (
    NOT_IN_DOCUMENTS,
    DraftAnswer,
    build_abstention_response,
    build_source_citations,
    format_verified_answer,
    generate_grounded_draft,
    verify_answer_claims,
)
from .llm import generate_from_evidence
from .paper_store import list_indexed_papers
from .question_planning import QuestionPlan, plan_question_tasks
from .quiz_validation import (
    QuizContent,
    QuizRequest,
    hide_quiz_answer_key,
    parse_quiz_command,
    validate_quiz_citations,
)
from .retrieval import CONTEXT_CHARACTERS, retrieve_relevant_passages


def assign_citation_labels(evidence: list[dict]) -> list[dict]:
    """Give selected evidence short source labels such as S1 and S2.

    Preserve each original chunk ID as chunk_id for tracing. Short labels are easier
    for Gemma to cite and for verification to look up within this answer.
    """

    return [
        {**row, "chunk_id": row["id"], "id": f"S{number}"}
        for number, row in enumerate(evidence, 1)
    ]


def generate_quiz_content(request: QuizRequest, evidence: list[dict]) -> dict:
    """Generate multiple-choice questions from retrieved paper evidence.

    Assign source labels, ask Gemma for structured questions and reject wrong
    counts, duplicate questions or unknown citations before returning the quiz
    content.
    """

    if not evidence:
        raise ValueError("No indexed paper evidence found for this quiz")
    evidence = assign_citation_labels(evidence)
    ids = ", ".join(row["id"] for row in evidence)
    quiz = generate_from_evidence(
        QuizContent,
        f"Create exactly {request.count} multiple-choice questions about the topic in the request. "
        f"Difficulty: {request.difficulty}. Easy=recall; medium=application; hard=scenarios. "
        f"Use only source IDs {ids}. Four distinct options, zero-based correct_index, "
        "an explanation, and source_ids for each question. Do not invent facts.",
        evidence,
        {"topic": request.topic},
        DEFAULT_OUTPUT_TOKENS,  # Up to ten questions with explanations need a longer output.
    )
    if len(quiz.questions) != request.count:
        raise ValueError("Wrong question count returned; try fewer questions")
    if (
        len({question.question.strip().casefold() for question in quiz.questions})
        != request.count
    ):
        raise ValueError("Duplicate quiz questions returned")
    for question in quiz.questions:
        validate_quiz_citations(question.source_ids, evidence)
    return {
        "topic": request.topic,
        "difficulty": request.difficulty,
        **quiz.model_dump(),
        "sources": build_source_citations(evidence),
    }


def create_and_save_quiz(request: QuizRequest) -> dict:
    """Retrieve topic evidence, generate a quiz and save its full answer key in SQLite.

    Return a quiz ID and questions without correct answers; submission uses the
    saved key for grading.
    """

    payload = generate_quiz_content(request, retrieve_relevant_passages(request.topic))
    quiz_id = create_record_id()
    with open_database() as database:
        database.execute(
            "INSERT INTO quizzes (id, payload) VALUES (?, ?)",
            (quiz_id, json.dumps(payload)),
        )
    return {"kind": "quiz", "quiz_id": quiz_id, **hide_quiz_answer_key(payload)}


WORKFLOW_VERSION = "reviewed-plan-v3"
# Each task still receives at least two short evidence windows within the shared budget.
MAX_TASKS_PER_BATCH = 5


class ChatState(TypedDict, total=False):
    """Describe the data passed between LangGraph nodes for one request.

    Nodes add a plan, evidence, a draft, verification and a final response to this
    shared dictionary. Optional fields track approval, deferred tasks and evaluation
    timings without requiring every route to generate an answer.
    """

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
    allow_clarification: bool
    consistency_report: NotRequired[dict]


def plan_search_tasks(state: ChatState) -> dict:
    """Prepare search tasks from an approved plan or a newly planned question.

    Check named papers against the current index and request review when needed.
    Execute only a batch that fits the response context budget, retaining remaining
    tasks for later approval. Return graph state updates and planning diagnostics.
    """
    started = perf_counter()
    papers = list_indexed_papers()
    indexed_titles = {row["title"] for row in papers}
    approved = state.get("approved_plan")
    planned = (
        QuestionPlan.model_validate(approved)
        if approved
        else plan_question_tasks(state["message"], state.get("history", []), papers)
    )
    if planned.clarification and not state.get("allow_clarification", True):
        # Unattended callers (API, evaluation) cannot answer a question back; search everything.
        planned = QuestionPlan(tasks=[SearchTask(question=state["message"])])
    if approved and any(
        task.paper and task.paper not in indexed_titles for task in planned.tasks
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
    # Batches exist so a reviewer can approve the next one; unattended runs do every task.
    batch_size = MAX_TASKS_PER_BATCH if state.get("require_review") else max(len(tasks), 1)
    deferred = tasks[batch_size:] if not review else []
    if not review:
        tasks = tasks[:batch_size]
    if approved:
        approval_mode = "user_approved"
    elif state.get("require_review"):
        approval_mode = "review_required"
    else:
        approval_mode = "automatic_offline"
    return {
        "tasks": tasks,
        "awaiting_approval": review,
        "deferred_tasks": deferred,
        "clarification": planned.clarification,
        "timings": {"planning": perf_counter() - started},
        "retrieval_trace": {
            "route": planned.route,
            "workflow": WORKFLOW_VERSION,
            "approval_mode": approval_mode,
            "planned_task_count": len(planned.tasks),
        },
    }


def choose_route_after_planning(state: ChatState) -> str:
    """Choose whether the graph can search or must return a clarification/approval request.

    This conditional edge prevents retrieval and generation from running before the
    question scope is ready.
    """
    return (
        "format_response"
        if state.get("clarification") or state.get("awaiting_approval")
        else "retrieve"
    )


def retrieve_task_evidence(state: ChatState) -> dict:
    """Retrieve evidence for each task and share one context budget across the tasks.

    Take passages in turns so one task does not consume all available space. Keep
    shared passages once while recording every task they address, then assign
    citation labels and collect retrieval diagnostics.
    """
    started = perf_counter()
    tasks = state["tasks"]
    groups, runs = [], []
    for task in tasks:
        trace: dict = {"capture_evidence": state.get("capture_evidence", False)}
        groups.append(
            retrieve_relevant_passages(
                task["question"], paper=task["paper"], diagnostics=trace
            )
        )
        runs.append({"task": task, "trace": trace})
    # A sub-question phrased unlike the paper can score below the threshold even when
    # its siblings found evidence; keep its two best passages instead of dropping it.
    if any(groups):
        for task_id, group in enumerate(groups):
            if not group:
                task = tasks[task_id]
                groups[task_id] = search(
                    task["question"], paper=task["paper"],
                    diagnostics=runs[task_id]["trace"], fallback=2,
                )
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
    passages = assign_citation_labels(list(evidence.values()))
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


def choose_route_after_retrieval(state: ChatState) -> str:
    """Generate an answer only when retrieval returned evidence.

    An empty evidence list goes directly to response formatting, which returns an
    explicit no-evidence response without asking Gemma to guess.
    """
    return "generate" if state["evidence"] else "format_response"


def generate_answer_draft(state: ChatState) -> dict:
    """Ask Gemma for factual claims using the question, short history and selected evidence.

    Store the structured draft in graph state and record generation time.
    Verification runs next before any claims are shown to the user.
    """
    started = perf_counter()
    history = [
        {"role": item["role"], "content": item["content"][:500]}
        for item in state.get("history", [])[-2:]
    ]
    draft = generate_grounded_draft(
        state["message"], history, state["evidence"], state["tasks"]
    )
    return {
        "draft": draft,
        "timings": {**state["timings"], "generation": perf_counter() - started},
        "consistency_report": consistency_report,
    }


def verify_answer_draft(state: ChatState) -> dict:
    """Check the draft's sources, quotes, numbers and task associations in Python.

    Store accepted claims and missing tasks for response formatting. This uses no
    additional model call and does not establish whether every paraphrase is
    logically supported.
    """
    started = perf_counter()
    checked = verify_answer_claims(state["draft"], state["evidence"], state["tasks"])
    return {
        "verification": checked,
        "timings": {**state["timings"], "verification": perf_counter() - started},
    }


def build_chat_response(state: ChatState) -> dict:
    """Convert the graph's current outcome into one browser-compatible response.

    Handle clarification, plan approval, missing evidence or verified claims. If an
    execution batch leaves tasks unfinished, attach a new proposal so the user can
    continue instead of silently losing those tasks.
    """
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
        result["plan"] = create_plan_proposal(
            state["message"], state["tasks"], state["retrieval_trace"]["route"]
        )
        result["answer"] += "\n\n" + "\n".join(
            f"{i + 1}. {task['question']}"
            + (f" — {task['paper']}" if task.get("paper") else "")
            for i, task in enumerate(state["tasks"])
        )
    elif not state.get("evidence"):
        result = build_abstention_response(
            "No matching research-paper evidence was found for this question."
        )
    else:
        result = format_verified_answer(
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
        result["plan"] = create_plan_proposal(
            state["message"], state["deferred_tasks"], state["retrieval_trace"]["route"]
        )
        result["answer"] += "\n\nRemaining tasks:\n" + "\n".join(
            f"{i + 1}. {task['question']}"
            + (f" — {task['paper']}" if task.get("paper") else "")
            for i, task in enumerate(state["deferred_tasks"])
        )
    return {"result": result}


def create_plan_proposal(question: str, tasks: list[dict], route: str) -> dict:
    """Attach a new ID to the exact question, task list and route proposed for review.

    The API saves this proposal and checks its ID on approval so an outdated browser
    action cannot execute a different plan.
    """
    return {
        "plan_id": create_record_id(),
        "question": question,
        "tasks": tasks,
        "route": route,
    }


# Each node returns updates to ChatState; conditional edges choose the next node.
# Plan -> clarify/review OR retrieve -> no-evidence response OR generate -> verify.
workflow = StateGraph(ChatState)
workflow.add_node("plan", plan_search_tasks)
workflow.add_node("retrieve", retrieve_task_evidence)
workflow.add_node("generate", generate_answer_draft)
workflow.add_node("verify", verify_answer_draft)
workflow.add_node("format_response", build_chat_response)
workflow.add_edge(START, "plan")
workflow.add_conditional_edges(
    "plan",
    choose_route_after_planning,
    {name: name for name in ("retrieve", "format_response")},
)
workflow.add_conditional_edges(
    "retrieve",
    choose_route_after_retrieval,
    {name: name for name in ("generate", "format_response")},
)
workflow.add_edge("generate", "verify")
workflow.add_edge("verify", "format_response")
workflow.add_edge("format_response", END)
chat_graph = workflow.compile()


def answer_research_question(
    message: str,
    history: list[dict] | None = None,
    capture_evidence: bool = True,
    require_review: bool = False,
    approved_plan: dict | None = None,
    force_review: bool = False,
    allow_clarification: bool = True,
) -> dict:
    """Run the research graph and return the answer, evidence, trace and timings.

    The browser can require plan review; offline evaluation uses automatic planning.
    Only the question and allowed conversation/plan inputs enter this function, not
    benchmark reference answers.
    """
    started = perf_counter()
    state = chat_graph.invoke(
        {
            "message": message,
            "history": history or [],
            "capture_evidence": capture_evidence,
            "require_review": require_review,
            "approved_plan": approved_plan,
            "force_review": force_review,
            "allow_clarification": allow_clarification,
        }
    )
    return {
        **state["result"],
        "context": state.get("evidence", []),
        "retrieval_trace": state.get("retrieval_trace", {}),
        "timings": {**state.get("timings", {}), "total": perf_counter() - started},
    }


def build_paper_catalog_response() -> dict:
    """Build the /papers answer from indexed document metadata.

    Return titles and source links directly, avoiding retrieval or a model-generated
    guess about which papers exist.
    """

    papers = list_indexed_papers()
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


def handle_chat_message(
    message: str, history: list[dict], force_review: bool = False
) -> dict:
    """Send /papers and /quiz commands to their handlers, otherwise run the research graph.

    Browser questions require review for complex plans. Remove raw evidence bodies
    from the browser response while retaining the answer and citation links.
    """

    if message.strip().casefold() == "/papers":
        return build_paper_catalog_response()
    if message.strip().startswith("/quiz"):
        return create_and_save_quiz(parse_quiz_command(message))
    result = answer_research_question(
        message,
        history,
        capture_evidence=False,
        require_review=True,
        force_review=force_review,
    )
    result.pop("context")
    return result
