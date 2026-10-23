-- Chroma stores PDF text, vectors and source metadata. SQLite stores UI state.
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One immutable summary per ingestion attempt makes retrieval results traceable.
CREATE TABLE IF NOT EXISTS index_runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    index_signature TEXT NOT NULL,
    rebuild INTEGER NOT NULL,
    source_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    counts TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    messages TEXT NOT NULL DEFAULT '[]'
);

-- A new proposal replaces the old token, so stale approvals cannot execute edits.
CREATE TABLE IF NOT EXISTS pending_plans (
    session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
    plan_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS quizzes (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id TEXT PRIMARY KEY,
    quiz_id TEXT NOT NULL REFERENCES quizzes(id),
    answers TEXT NOT NULL,
    score INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_challenges (
    day TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    answer INTEGER
);
