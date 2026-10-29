"""Build explicit per-detail model fixtures from short cited example paragraphs."""

import re

from app.grounding import DetailAnswer, DraftAnswer


def cited_draft(answer: str = "") -> DraftAnswer:
    """Use one example paragraph per requested detail, with source labels separated."""
    return DraftAnswer(
        answers=[
            DetailAnswer(
                availability="available",
                answer=re.sub(r"\[[^\]]*\bS\d+\b[^\]]*\]", "", part).strip(),
                source_ids=re.findall(r"\bS\d+\b", part),
            )
            for part in re.split(r"\n\s*\n", answer)
            if part.strip()
        ]
    )
