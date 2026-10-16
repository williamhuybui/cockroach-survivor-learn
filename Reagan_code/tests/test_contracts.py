from datetime import date
import pytest
from pydantic import ValidationError
from app.schemas import Question, QuizRequest, grade_quiz, last_week, parse_quiz_command, public_quiz, validate_source_ids
from app.ingest import parse_actions


def question():
    return dict(question="Which component finds similar vectors?", options=["pgvector", "CSS", "HTML", "JWT"],
                correct_index=0, explanation="pgvector supports similarity search", source_ids=["source-1"])


@pytest.mark.parametrize("count", [0, 11, -1, 3.5])
def test_quiz_limit(count):
    with pytest.raises(ValidationError):
        QuizRequest(topic="RAG", count=count)


def test_command_with_multiple_words():
    req = parse_quiz_command('/quiz "database migration" hard 10')
    assert (req.topic, req.difficulty, req.count) == ("database migration", "hard", 10)


def test_duplicate_options_rejected():
    q = question()
    q["options"] = ["SQL", " sql ", "HTML", "CSS"]
    with pytest.raises(ValidationError):
        Question(**q)


def test_unknown_evidence_rejected():
    with pytest.raises(ValueError):
        validate_source_ids(["invented"], [{"id": "source-1"}])



def test_last_week_across_year():
    assert last_week(date(2026, 1, 1)) == (date(2025, 12, 22), date(2025, 12, 28))


def test_public_quiz_does_not_leak_answer_key():
    result = public_quiz({"topic": "RAG", "difficulty": "easy", "questions": [question()]})
    assert set(result["questions"][0]) == {"question", "options"}


def test_grading_rejects_wrong_length_and_range():
    assert grade_quiz([question()], [0]) == 1
    assert grade_quiz([question()], [1]) == 0
    for answers in ([], [4], [-1], [True]):
        with pytest.raises(ValueError):
            grade_quiz([question()], answers)


def test_action_table_preserves_exact_owner_status():
    text = "| Action | Owner | Status | Due |\n| --- | --- | --- | --- |\n| Review code | Tri | open | 2026-10-09 |"
    result = parse_actions(text)
    assert result[0]["owner"] == "Tri"
    assert result[0]["due_date"] == date(2026, 10, 9)
    assert result[0]["location"] == "line 3"


def test_invalid_action_status_is_not_silently_guessed():
    with pytest.raises(ValueError):
        parse_actions("| Action | Owner | Status | Due |\n| Review | Tri | perhaps | |")
