"""Command-line setup for local chat and quiz storage."""

import argparse

from .db import initialize_database


def main():
    """Create the small SQLite tables; Chroma initializes during ingestion."""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init-db"])
    parser.parse_args()
    initialize_database()
    print("Local chat and quiz database is ready.")


if __name__ == "__main__":
    main()
