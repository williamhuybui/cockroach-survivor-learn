import shlex
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, StrictInt, model_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class QuizRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    count: int = Field(default=5, ge=1, le=10, strict=True)




class SubmitRequest(BaseModel):
    answers: list[StrictInt] = Field(min_length=1, max_length=10)


class Answer(BaseModel):
    answer: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)


class Question(BaseModel):
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3, strict=True)
    explanation: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_options(self):
        """Reject blank or duplicate choices before storing a question."""
        if len({x.strip().casefold() for x in self.options}) != 4 or any(not x.strip() for x in self.options):
            raise ValueError("Four distinct non-empty options are required")
        return self


class QuizContent(BaseModel):
    questions: list[Question] = Field(min_length=1, max_length=10)








def validate_source_ids(ids, evidence):
    """Reject citations that were not present in the retrieved evidence."""
    allowed = {e["id"] for e in evidence}
    if not ids or not set(ids).issubset(allowed):
        raise ValueError("Model returned a citation outside the retrieved evidence")


def parse_quiz_command(message):
    """Parse a topic, difficulty, and question count from an explicit quiz command."""
    parts = shlex.split(message)
    if len(parts) < 4 or parts[0] != "/quiz":
        raise ValueError('Use /quiz "topic" easy|medium|hard 1-10')
    return QuizRequest(topic=" ".join(parts[1:-2]), difficulty=parts[-2], count=int(parts[-1]))


def last_week(today: date):
    """Return the previous Monday through Sunday, including across year boundaries."""
    this_monday = today - timedelta(days=today.weekday())
    return this_monday - timedelta(days=7), this_monday - timedelta(days=1)


def public_quiz(payload):
    """Strip answers, explanations, and sources until the member submits."""
    return {"topic": payload["topic"], "difficulty": payload["difficulty"],
            "questions": [{"question": q["question"], "options": q["options"]} for q in payload["questions"]]}


def grade_quiz(questions, answers):
    """Validate one answer per question and count correct choices without a model."""
    if len(questions) != len(answers) or any(type(a) is not int or a < 0 or a > 3 for a in answers):
        raise ValueError("Provide one answer index from 0 to 3 for every question")
    return sum(q["correct_index"] == a for q, a in zip(questions, answers))
