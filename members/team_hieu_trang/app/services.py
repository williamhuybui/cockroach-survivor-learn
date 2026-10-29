"""Coordinate the question-to-answer workflow using LangGraph.

The graph plans search tasks, retrieves evidence, asks Gemma for a draft, verifies
its references and formats a response. Conditional edges stop for clarification,
plan approval or missing evidence. Chat commands and quizzes enter through
handle_chat_message().
"""

import json
from time import perf_counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .db import create_record_id, open_database
from .grounding import (
    DraftAnswer,
    build_abstention_response,
    build_source_citations,
    format_cited_answer,
    generate_grounded_draft,
    verify_answer_citations,
)
from .llm import generate_from_evidence
from .paper_store import list_indexed_papers
from .question_planning import (
    QuestionPlan,
    plan_question_tasks,
)
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


WORKFLOW_VERSION = "separate-paper-search-v15"


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


def plan_search_tasks(state: ChatState) -> dict:
    """Prepare search tasks from an approved plan or a newly planned question.

    Check named papers against the current index and request review when needed.
    Execute only a batch that fits the response context budget, retaining remaining
    tasks for later approval in the browser. Automatic callers receive explicit
    gaps instead of a continuation proposal. Return planning diagnostics.
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
    if approved:
        for task in planned.tasks:
            selected = [task.paper] if isinstance(task.paper, str) else task.paper or []
            if not set(selected + task.candidate_papers) <= indexed_titles:
                planned = QuestionPlan(
                    clarification="The indexed papers changed. Please revise the plan before approving it again."
                )
                break
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
    deferred, executable = [], []
    searches = 0
    if not review:
        for task in tasks:
            if task.get("scope") == "unresolved":
                executable.append(task)
            elif searches < batch_size:
                executable.append(task)
                searches += 1
            else:
                deferred.append(task)
        tasks = executable
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
            "execution_task_limit": batch_size,
            "executed_task_count": 0 if review or planned.clarification else searches,
            "unresolved_task_count": sum(
                task.get("scope") == "unresolved" for task in tasks
            ),
            "unexecuted_task_count": len(deferred),
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
    tasks = [dict(task) for task in state["tasks"]]
    # For several parts of the same paper, the original request supplies any
    # subject that a short model-generated task accidentally left out.
    same_paper = (
        len(tasks) > 1
        and isinstance(tasks[0].get("paper"), str)
        and all(task.get("paper") == tasks[0]["paper"] for task in tasks)
    )
    groups, runs = [], []
    for task in tasks:
        trace: dict = {"capture_evidence": state.get("capture_evidence", False)}
        if task.get("scope") == "unresolved":
            groups.append([])
            runs.append({"task": task, "trace": {**trace, "scope_unresolved": True}})
            continue
        query = task["question"]
        if same_paper and state.get("message") and tasks[0]["paper"] not in query:
            query += f"\nQuestion context: {state['message']}"
        passages = retrieve_relevant_passages(
            query,
            paper=task["paper"] or task.get("candidate_papers") or None,
            diagnostics=trace,
            **({"resolve_candidates": True} if task.get("candidate_papers") else {}),
        )
        if task.get("candidate_papers"):
            task["paper"] = trace.get("resolved_paper")
            if not task["paper"]:
                task["scope"] = "unresolved"
                passages = []
        if len(tasks) > 1 and passages:
            # A shared budget may otherwise drop a strong keyword match behind
            # several similar semantic hits. Keep MiniLM's top result first and
            # give the best accepted BM25 match the next opportunity for context.
            keyword = max(passages, key=lambda row: row.get("bm25_score", 0))
            if keyword["id"] != passages[0]["id"] and keyword.get("bm25_score", 0) > 0:
                passages = [passages[0], keyword] + [
                    row for row in passages[1:] if row["id"] != keyword["id"]
                ]
        groups.append(passages)
        runs.append({"task": task, "trace": trace})
    # Round-robin selection gives each task a chance before spending the budget.
    remaining, evidence = CONTEXT_CHARACTERS, {}
    task_budget = CONTEXT_CHARACTERS // max(1, sum(bool(group) for group in groups))
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
    # Keep each task's selected passages together before assigning source labels.
    # Gemma sees contiguous labels for each task, while shared evidence stays once.
    ordered = sorted(evidence.values(), key=lambda row: min(row["task_ids"]))
    passages = assign_citation_labels(ordered)
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
        "tasks": tasks,
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
    """Ask Gemma once for answers to the planner's standalone requested details.

    Store the structured draft in graph state and record generation time.
    Citation checks run next before the answer is shown to the user.
    """
    started = perf_counter()
    # The planner already resolved follow-ups into standalone detail questions.
    draft = generate_grounded_draft(state["message"], state["evidence"], state["tasks"])
    return {
        "draft": draft,
        "timings": {**state["timings"], "generation": perf_counter() - started},
    }


def verify_answer_draft(state: ChatState) -> dict:
    """Validate answer citations and numeric values once, without another model call."""
    started = perf_counter()
    checked = verify_answer_citations(
        state["draft"], state["evidence"], state["tasks"], question=state["message"]
    )
    return {
        "verification": checked,
        "timings": {**state["timings"], "verification": perf_counter() - started},
        "retrieval_trace": {
            **state.get("retrieval_trace", {}),
            "citation_checks": {
                "accepted_detail_count": len(checked["accepted"]),
                "rejected_count": checked["rejected_count"],
                "rejections": checked["rejections"],
            },
        },
    }


def build_chat_response(state: ChatState) -> dict:
    """Convert the graph's current outcome into one browser-compatible response.

    Browser requests can return clarification or an approval proposal. Automatic
    requests return an answer, partial answer or abstention, including unfinished
    tasks without asking for another turn.
    """
    interactive = state.get("require_review", False)
    if state.get("clarification") and not interactive:
        # An independent evaluation request has nobody available to resolve ambiguity.
        result = build_abstention_response(
            "I don't know - the requested paper or question scope could not be resolved from the indexed documents."
        )
        result["abstention_reason"] = "unresolved_scope"
    elif state.get("clarification"):
        result = {
            "kind": "answer",
            "status": "clarification",
            "answer": state["clarification"],
            "sources": [],
        }
    elif state.get("awaiting_approval"):
        result = {
            "kind": "answer",
            "status": "awaiting_approval",
            "answer": "Please review this plan. Approve it to start searching, or describe what to revise.",
            "sources": [],
        }
        result["plan"] = create_plan_proposal(
            state["message"], state["tasks"], state["retrieval_trace"]["route"]
        )
        result["answer"] += "\n\n" + "\n".join(
            f"{i + 1}. {task['question']}"
            + (f" â€” {task['paper']}" if task.get("paper") else "")
            for i, task in enumerate(state["tasks"])
        )
    elif not state.get("evidence"):
        ambiguous = any(
            run["trace"].get("scope_ambiguous")
            for run in state.get("retrieval_trace", {}).get("tasks", [])
        )
        result = build_abstention_response(
            "Several indexed papers remain plausible; the evidence does not identify a single requested paper."
            if ambiguous
            else "No matching research-paper evidence was found for this question."
        )
    else:
        result = format_cited_answer(
            state["verification"], state["evidence"], state["tasks"]
        )
    if not interactive and result["status"] == "abstained":
        result["missing_tasks"] = [task["question"] for task in state.get("tasks", [])]
    if state.get("deferred_tasks"):
        if not state.get("evidence"):
            result["answer"] = (
                "No matching evidence was found for the tasks in this execution batch."
            )
        if interactive:
            result["answer"] += (
                f"\n\nThis response covers one execution batch. {len(state['deferred_tasks'])} search tasks remain; approve the next batch to continue."
            )
        else:
            result["answer"] += (
                "\n\nThe execution budget was reached. The following parts were not searched and are unanswered."
            )
        result["status"] = "partial" if result.get("sources") else result["status"]
        if interactive:
            result["plan"] = create_plan_proposal(
                state["message"],
                state["deferred_tasks"],
                state["retrieval_trace"]["route"],
            )
        else:
            result["unexecuted_tasks"] = [
                task["question"] for task in state["deferred_tasks"]
            ]
            result["missing_tasks"] = list(
                dict.fromkeys(
                    result.get("missing_tasks", []) + result["unexecuted_tasks"]
                )
            )
        result["answer"] += "\n\nUnfinished parts:\n" + "\n".join(
            f"{i + 1}. {task['question']}"
            + (f" â€” {task['paper']}" if task.get("paper") else "")
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
) -> dict:
    """Run the research graph and return the answer, evidence, trace and timings.

    The browser can require plan review. With require_review=False, execute the
    planned task batch automatically, abstain on unresolved scope and report any
    parts beyond the execution budget without asking for approval or clarification.
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
