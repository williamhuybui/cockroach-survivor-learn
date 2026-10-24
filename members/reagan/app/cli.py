import argparse
from sqlalchemy import text
from .db import Base, SHARED_USER_ID, Session, User, engine


def main():
    """Create the database objects required by the shared team workspace."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init-db"])
    args = parser.parse_args()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with Session.begin() as db:
        # Existing foreign keys still expect a users row, even though the UI has no accounts.
        if db.get(User, SHARED_USER_ID) is None:
            db.add(User(id=SHARED_USER_ID, username=SHARED_USER_ID, password_hash="login-disabled"))
    with engine.begin() as connection:
        # Expression index works on existing chunks without adding a generated column.
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS chunks_text_idx ON document_chunks "
            "USING gin (to_tsvector('simple', content))"
        ))
    print("Database ready. Added missing tables, shared workspace, and keyword index.")


if __name__ == "__main__":
    main()
