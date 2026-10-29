-- Local SQLite tables opened by db.py for conversations, quizzes and indexing history.
-- PDF passages, section text and embedding vectors are stored separately in Chroma.
-- CREATE IF NOT EXISTS prepares missing tables without deleting existing application records.

-- Small application values, including the last successful ingestion timestamp.
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One summary per indexing attempt: source/settings hashes, counts, outcome and times.
-- These records explain how an index was built; they do not back up its vectors.
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

-- Conversation messages restored by the browser and used as follow-up context.
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    messages TEXT NOT NULL DEFAULT '[]'
);

-- Current reviewable search plan per chat, with its ID and pending/running state.
-- Removing a chat also removes its pending plan; stale plan IDs cannot approve new content.
CREATE TABLE IF NOT EXISTS pending_plans (
    session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
    plan_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
);

-- Full generated quizzes, including answer keys kept server-side until submission.
CREATE TABLE IF NOT EXISTS quizzes (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

-- Submitted option indexes and scores for each attempt at a saved quiz.
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id TEXT PRIMARY KEY,
    quiz_id TEXT NOT NULL REFERENCES quizzes(id),
    answers TEXT NOT NULL,
    score INTEGER NOT NULL
);

-- One saved question and first submitted option per team calendar date.
CREATE TABLE IF NOT EXISTS daily_challenges (
    day TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    answer INTEGER
);
