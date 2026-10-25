
import pytest

from app.question_planning import check_history_relevance


@pytest.mark.parametrize(
    "previous,current,expected",
    [
        (
            "Explain RAG-Token.",
            "How does it work?",
            True,
        ),
        (
            "Explain Transformers.",
            "What is BM25?",
            False,
        ),
        (
            "Compare RAG-Token and RAG-Sequence.",
            "Which one performs better?",
            True,
        ),
        (
            "Explain attention mechanisms.",
            "What is the weather tomorrow?",
            False,
        ),
        (
            "Explain RAG-Sequence.",
            "What are its limitations?",
            True,
        ),
        (
            "What is BM25?",
            "Explain Transformer architecture.",
            False,
        ),
    ],
)
def test_history_relevance(previous, current, expected):
    history = [
        {"role": "human", "content": previous},
    ]

    result = check_history_relevance(current, history)

    assert result["use_history"] == expected
