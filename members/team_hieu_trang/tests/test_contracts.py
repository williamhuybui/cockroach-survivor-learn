"""Test quiz request validation, command parsing and answer-key handling.

Malformed counts, choices, citations and submissions must fail before a quiz is
stored or graded.
"""

import pytest
from pydantic import ValidationError

from app.quiz_validation import (
    Question,
    QuizRequest,
    grade_quiz,
    hide_quiz_answer_key,
    parse_quiz_command,
    validate_quiz_citations,
)


def question():
    return {
        "question": "Which component finds similar vectors?",
        "options": ["Chroma", "CSS", "HTML", "JWT"],
        "correct_index": 0,
        "explanation": "Chroma supports similarity search",
        "source_ids": ["source-1"],
    }


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
        validate_quiz_citations(["invented"], [{"id": "source-1"}])


def test_public_quiz_does_not_leak_answer_key():
    result = hide_quiz_answer_key(
        {"topic": "RAG", "difficulty": "easy", "questions": [question()]}
    )
    assert set(result["questions"][0]) == {"question", "options"}


def test_grading_rejects_wrong_length_and_range():
    assert grade_quiz([question()], [0]) == 1
    assert grade_quiz([question()], [1]) == 0
    for answers in ([], [4], [-1], [True]):
        with pytest.raises(ValueError):
            grade_quiz([question()], answers)
