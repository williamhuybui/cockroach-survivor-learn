"""Enrich the read-only paper catalog and resolve references without model guesses.

Aliases identify a paper's primary model or method, not every model it mentions.
Titles and filenames identify papers directly; aliases are optional. Weaker names
produce candidates for retrieval. SQLite connections close before model calls.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

from .db import initialize_database, open_database

# Primary names checked against paper titles/opening text, not evaluation labels.
# The local table can override these defaults, including replacing them with [].
INITIAL_ALIASES = json.loads(
    Path(__file__).with_name("catalog_aliases.json").read_text(encoding="utf-8")
)


def normalize_paper_reference(text: str) -> str:
    """Ignore formatting while keeping decimal versions such as 3.5 distinct from 3."""
    text = unicodedata.normalize("NFKC", text).casefold()
    # Remove sentence/filename dots, but keep dots between version digits.
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return re.sub(r"[^\w.]+|_", " ", text).strip()


def catalog_identity_names(row: dict) -> list[str]:
    """Read full names and recognizable leading identifiers directly from the catalog.

    A filename such as 05_ORION-3_Few_Shot_Learners.pdf already supplies ORION-3.
    Uppercase identifiers and labels before a colon also stand alone. Aliases are
    optional additions; no question-specific mapping is stored here.
    """
    stem = re.sub(r"^\d+[ _-]*", "", Path(row["title"]).stem)
    title = row.get("catalog_title", "")
    names = [row["title"], stem, title, *row.get("aliases", [])]
    for text in (stem.replace("_", " "), title):
        if not text:
            continue
        lead = text.split()[0].rstrip(":")
        if lead[0].isalpha() and (lead.isupper() or any(c.isdigit() for c in lead)):
            names.extend([lead, re.sub(r"(?<=[A-Za-z])(?=\d)", " ", lead)])
            names.append(normalize_paper_reference(lead).replace(" ", ""))
        if ":" in text:
            names.append(text.partition(":")[0])
    return names


def refresh_local_catalog(papers: list[dict]) -> None:
    """Copy current index/CSV identity fields while preserving verified aliases.

    Run during catalog maintenance, not on each question. Papers no longer indexed
    become inactive; their aliases remain stored for a later return of that document.
    """
    initialize_database()
    with open_database() as database:
        database.execute("UPDATE paper_catalog SET active = 0 WHERE active != 0")
        database.executemany(
            """INSERT INTO paper_catalog(document_id, paper_number, title, pdf_filename, phase, aliases, active)
               VALUES (?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT(document_id) DO UPDATE SET
                 paper_number=excluded.paper_number, title=excluded.title,
                 pdf_filename=excluded.pdf_filename, phase=excluded.phase, active=1""",
            [
                (
                    row["id"],
                    row.get("paper_number"),
                    row.get("catalog_title") or row["title"],
                    row["title"],
                    row.get("phase", ""),
                    json.dumps(INITIAL_ALIASES.get(row["title"], [])),
                )
                for row in papers
            ],
        )


def load_local_aliases() -> dict[str, list[str]]:
    """Read aliases in one short transaction; a not-yet-initialized catalog is empty."""
    with open_database() as database:
        if not database.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_catalog'"
        ).fetchone():
            return {}
        return {
            row["document_id"]: json.loads(row["aliases"])
            for row in database.execute(
                "SELECT document_id, aliases FROM paper_catalog WHERE active = 1"
            )
        }


def set_paper_aliases(document_id: str, aliases: list[str]) -> None:
    """Save verified primary names without editing index.csv or adding new document IDs.

    Reject duplicate ownership rather than silently assigning an ambiguous alias.
    This maintenance function is never called by the question planner or the LLM.
    """
    cleaned = list(dict.fromkeys(alias.strip() for alias in aliases))
    if any(len(normalize_paper_reference(alias)) < 3 for alias in cleaned):
        raise ValueError("Aliases need at least three normalized characters")
    with open_database() as database:
        rows = database.execute(
            "SELECT * FROM paper_catalog WHERE active = 1"
        ).fetchall()
        if document_id not in {row["document_id"] for row in rows}:
            raise ValueError("Alias owner must be an indexed catalog paper")
        for row in rows:
            if row["document_id"] == document_id:
                continue
            names = catalog_identity_names(
                {
                    "title": row["pdf_filename"],
                    "catalog_title": row["title"],
                    "aliases": json.loads(row["aliases"]),
                }
            )
            if {normalize_paper_reference(name) for name in names} & {
                normalize_paper_reference(alias) for alias in cleaned
            }:
                raise ValueError("Alias already identifies another paper")
        database.execute(
            "UPDATE paper_catalog SET aliases = ? WHERE document_id = ?",
            (json.dumps(cleaned), document_id),
        )


def find_paper_references(question: str, papers: list[dict]) -> dict:
    """Find clear catalog identities and weaker candidates worth retrieving.

    Match longer overlapping names first so 'Orion Extended' does not also match
    'Orion'. Separate mentions can still identify several papers. An alias with
    multiple owners remains ambiguous. Title fragments produce candidate papers,
    not permission to answer before their evidence has been examined.
    """
    query = " " + normalize_paper_reference(question) + " "
    owners: dict[str, set[str]] = {}
    for row in papers:
        for name in catalog_identity_names(row):
            name = normalize_paper_reference(name)
            if len(name) >= 3:
                owners.setdefault(name, set()).add(row["title"])
    titles, candidates, occupied, ambiguous = set(), set(), [], False
    for name in sorted(owners, key=len, reverse=True):
        # Whole phrases avoid matching a short name inside an unrelated word.
        for match in re.finditer(rf"(?<= ){re.escape(name)}(?= )", query):
            if any(
                match.start() < end and start < match.end() for start, end in occupied
            ):
                continue
            occupied.append(match.span())
            if len(owners[name]) == 1:
                titles.update(owners[name])
            else:
                ambiguous = True
                candidates.update(owners[name])
    # Bare numbers may be years or model sizes; only a labeled paper number counts.
    for number in re.findall(r"\bpaper\s*#?\s*(\d+)\b", question, re.IGNORECASE):
        matches = {
            row["title"] for row in papers if row.get("paper_number") == int(number)
        }
        titles.update(matches)
        ambiguous |= len(matches) > 1
    # An author may have written several papers. Treat author matches as search
    # candidates, never as proof that one paper is the requested one.
    author_candidates = set()
    first_author_candidates = set()
    if not titles:
        for row in papers:
            for index, author in enumerate(row.get("authors", [])):
                full_name = normalize_paper_reference(author)
                surname = full_name.split()[-1] if full_name else ""
                et_al = len(surname) >= 4 and re.search(
                    rf"\b{re.escape(surname)}\s+et\s+al\b", query
                )
                if (len(full_name.split()) > 1 and f" {full_name} " in query) or (
                    len(surname) >= 4
                    and (
                        et_al
                        or re.search(rf"\b{re.escape(surname.title())}\b", question)
                    )
                ):
                    author_candidates.add(row["title"])
                    if index == 0 and et_al:
                        first_author_candidates.add(row["title"])
                    break
        if author_candidates:
            # Title fragments can match unrelated papers; prefer the explicit author.
            # "Smith et al." conventionally names the first author, if known.
            candidates = first_author_candidates or author_candidates
    if not titles and not author_candidates:
        # Generic title words alone do not identify a paper. A distinctive leading
        # word or a multiword fragment can still supply candidates for retrieval.
        common = {
            "a",
            "an",
            "the",
            "of",
            "in",
            "on",
            "for",
            "and",
            "to",
            "with",
            "from",
            "by",
            "language",
            "models",
            "model",
            "paper",
            "learning",
            "training",
        }
        terms = set(normalize_paper_reference(question).split()) - common
        for row in papers:
            stem = re.sub(r"^\d+[ _-]*", "", Path(row["title"]).stem)
            words = normalize_paper_reference(stem).split()
            title_terms = (
                set(
                    normalize_paper_reference(
                        stem + " " + row.get("catalog_title", "")
                    ).split()
                )
                - common
            )
            # A numbered model's base token alone (ORION) must not identify an
            # unregistered version (ORION-3.5) through an ORION-3 filename.
            lead = stem.split("_")[0]
            distinct_lead = (
                words and len(words[0]) >= 4 and not any(c.isdigit() for c in lead)
            )
            if len(terms & title_terms) >= 2 or (distinct_lead and words[0] in terms):
                candidates.add(row["title"])
    return {
        "titles": sorted(titles),
        "candidates": sorted(candidates - titles),
        "ambiguous": ambiguous,
    }


if __name__ == "__main__":
    from .paper_store import list_indexed_papers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-id",
        help="Existing indexed ID whose verified aliases should be replaced",
    )
    parser.add_argument("--alias", action="append", default=[])
    args = parser.parse_args()
    refresh_local_catalog(list_indexed_papers())
    if args.document_id:
        set_paper_aliases(args.document_id, args.alias)
    print(json.dumps(list_indexed_papers(), indent=2))
