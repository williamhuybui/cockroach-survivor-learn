"""Check the shared catalog-constrained planner without question-specific branches."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from app import question_planning as planning


@pytest.mark.parametrize(
    "question",
    [
        "Explain Orion 8x4B",
        "What is Orion?",
        "List Orion assumptions and limitations",
        "What does paper 42 say?",
    ],
)
def test_all_wordings_use_one_catalog_contract(monkeypatch, question):
    """Model names, catalog numbers and multipart wording share a validated interface."""
    calls = []

    def plan(schema, instruction, payload):
        """Select an allowed filename, never normalize a guessed model output later."""
        calls.append(payload)
        with pytest.raises(ValidationError):
            schema(tasks=[{"question": "Orion?", "paper": "42", "scope": "paper"}])
        with pytest.raises(ValidationError):
            schema(tasks=[{"question": "Orion?", "scope": "paper"}])
        return schema(
            tasks=[{"question": question, "paper": "42_Orion.pdf", "scope": "paper"}]
        )

    monkeypatch.setattr(planning, "request_structured_response", plan)
    result = planning.plan_question_tasks(
        question, [], [{"title": "42_Orion.pdf", "paper_number": 42}]
    )
    assert len(calls) == 1
    assert result.tasks[0].paper == "42_Orion.pdf"


def test_request_local_catalogs_do_not_leak_between_threads(monkeypatch):
    """Schema construction does not mutate a shared enum or block other questions."""

    def plan(schema, instruction, payload):
        """Return a plan that validates against this call's own catalog."""
        title = payload["indexed_papers"][0]["title"]
        return schema(
            tasks=[{"question": payload["question"], "paper": title, "scope": "paper"}]
        )

    monkeypatch.setattr(planning, "request_structured_response", plan)

    def run(index):
        """Use a distinct indexed filename for each simultaneous request."""
        return (
            planning.plan_question_tasks(
                f"Explain Paper_{index}.pdf", [], [{"title": f"Paper_{index}.pdf"}]
            )
            .tasks[0]
            .paper
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run, range(4))) == [f"Paper_{i}.pdf" for i in range(4)]
