"""Define and validate the quiz data shared by generation, API routes and grading.

Pydantic checks request and model-output shapes. Helpers parse /quiz commands, hide
answer keys until submission, validate citations and grade option indexes against
saved correct answers.
"""

import shlex
from typing import Literal

from pydantic import BaseModel, Field, StrictInt, model_validator


# A parsed /quiz command becomes this validated topic, difficulty and count.
# Generation uses these settings after retrieving research-paper evidence.
class QuizRequest(BaseModel):
    """A requested topic, difficulty, and at most ten questions."""

    topic: str = Field(min_length=1, max_length=200)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    count: int = Field(default=5, ge=1, le=10, strict=True)


# Browser quiz submissions contain one option index per question.
# Index 0 means the first choice; the saved server-side key determines the score.
class SubmitRequest(BaseModel):
    """Zero-based option indices selected by the user."""

    answers: list[StrictInt] = Field(min_length=1, max_length=10)


# Gemma must return question text, four choices, a correct index and citations.
# The full record is saved on the server; answer fields are hidden before submission.
class Question(BaseModel):
    """One multiple-choice question with a server-side answer key."""

    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3, strict=True)
    explanation: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct_options(self):
        """Reject blank or repeated answer choices after Pydantic checks their count.

        Compare trimmed, case-insensitive text so four differently formatted copies
        are not accepted as distinct options.
        """
        options = [option.strip().casefold() for option in self.options]
        if "" in options or len(set(options)) != 4:
            raise ValueError("Four distinct non-empty options are required")
        return self


# Group generated questions under one schema for structured model output.
# Further generation checks enforce the requested count and valid source labels.
class QuizContent(BaseModel):
    """A structured batch of questions returned by Gemma 3."""

    questions: list[Question] = Field(min_length=1, max_length=10)


def validate_quiz_citations(ids, evidence):
    """Check that every generated source label belongs to the supplied quiz evidence.

    Reject missing or invented labels before the generated question is saved.
    """
    allowed = {e["id"] for e in evidence}
    if not ids or not set(ids).issubset(allowed):
        raise ValueError("Model returned a citation outside the retrieved evidence")


def parse_quiz_command(message):
    """Turn /quiz "topic" difficulty count into a validated QuizRequest.

    Use shell-style splitting to preserve quoted topics, then let the request model
    check allowed difficulty and question count.
    """
    parts = shlex.split(message)
    if len(parts) < 4 or parts[0] != "/quiz":
        raise ValueError('Use /quiz "topic" easy|medium|hard 1-10')
    return QuizRequest(
        topic=" ".join(parts[1:-2]), difficulty=parts[-2], count=int(parts[-1])
    )


def hide_quiz_answer_key(payload):
    """Build the pre-submission quiz response from the saved full quiz.

    Expose question text and choices while keeping correct answers, explanations and
    citation details server-side until grading.
    """
    return {
        "topic": payload["topic"],
        "difficulty": payload["difficulty"],
        "questions": [
            {"question": q["question"], "options": q["options"]}
            for q in payload["questions"]
        ],
    }


def grade_quiz(questions, answers):
    """Validate one selected option per question and return the number correct.

    Choices are integer indexes 0 through 3. Compare with the stored answer key
    rather than asking a model to judge the submission.
    """
    if len(questions) != len(answers) or any(
        type(a) is not int or a < 0 or a > 3 for a in answers
    ):
        raise ValueError("Provide one answer index from 0 to 3 for every question")
    return sum(q["correct_index"] == a for q, a in zip(questions, answers))
