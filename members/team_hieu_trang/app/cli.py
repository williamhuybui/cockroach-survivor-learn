"""Command-line setup for the application's local SQLite storage.

Paper vectors are created separately by app.ingest; this module prepares chat, quiz
and history tables.
"""

import argparse

from .db import initialize_database


def main():
    """Run the setup command and create missing SQLite tables.

    This initializes application records, not paper extraction, embeddings or Chroma
    ingestion.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init-db"])
    parser.parse_args()
    initialize_database()
    print("Local chat and quiz database is ready.")


if __name__ == "__main__":
    main()
