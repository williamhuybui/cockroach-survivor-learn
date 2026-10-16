from datetime import date, datetime, timezone
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import DATABASE_URL, EMBED_DIM

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 5, "options": "-c statement_timeout=15000"})
Session = sessionmaker(engine, expire_on_commit=False)
SHARED_USER_ID = "team-workspace"


def new_id():
    """Create a unique ID for a user, chat, quiz, or quiz attempt."""
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String)
    source_url: Mapped[str] = mapped_column(Text)
    author: Mapped[str] = mapped_column(String, default="")
    topic: Mapped[str] = mapped_column(String, default="")
    session_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    fingerprint: Mapped[str] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Chunk(Base):
    __tablename__ = "document_chunks"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    content: Mapped[str] = mapped_column(Text)
    location: Mapped[str] = mapped_column(String)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM))


class ActionItem(Base):
    __tablename__ = "action_items"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    description: Mapped[str] = mapped_column(Text)
    owner: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, index=True)
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    location: Mapped[str] = mapped_column(String)


class User(Base):
    """Internal shared identity retained for existing database foreign keys."""

    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String, unique=True)
    password_hash: Mapped[str] = mapped_column(Text)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), default=SHARED_USER_ID, index=True)
    messages: Mapped[list] = mapped_column(JSONB, default=list)


class Quiz(Base):
    __tablename__ = "quizzes"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    payload: Mapped[dict] = mapped_column(JSONB)


class Attempt(Base):
    __tablename__ = "quiz_attempts"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    quiz_id: Mapped[str] = mapped_column(ForeignKey("quizzes.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    answers: Mapped[list] = mapped_column(JSONB)
    score: Mapped[int] = mapped_column(Integer)


class DailyChallenge(Base):
    """One shared question per team-local date; answers stay on the server."""

    __tablename__ = "daily_challenges"
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSONB)


class DailyAttempt(Base):
    """Keep each member's first daily answer without changing other members' results."""

    __tablename__ = "daily_attempts"
    day: Mapped[date] = mapped_column(ForeignKey("daily_challenges.day"), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    answer: Mapped[int] = mapped_column(Integer)

