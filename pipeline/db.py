"""SQLite storage. Uses only the Python standard library."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS generation_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL DEFAULT (datetime('now')),
    model        TEXT NOT NULL,
    seed         INTEGER,
    requested    INTEGER NOT NULL,
    notes        TEXT
);

-- Synthetic tickets plus the labels they were generated FROM (ground truth).
CREATE TABLE IF NOT EXISTS tickets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES generation_runs(id),
    text            TEXT NOT NULL UNIQUE,
    channel         TEXT NOT NULL,          -- email | chat | app_review
    lang            TEXT NOT NULL,          -- en, zh, ja, ko, es, pt, fr, de, hi
    sentiment       TEXT NOT NULL,          -- positive | neutral | negative
    department      TEXT NOT NULL,          -- billing | technical | account | sales | other
    urgency         INTEGER NOT NULL,       -- 0 not urgent, 1 soon, 2 blocking
    churn_risk      INTEGER NOT NULL,       -- 0/1
    refund_request  INTEGER NOT NULL,       -- 0/1
    difficulty      TEXT NOT NULL,          -- plain | sarcasm | mixed | implicit_churn | code_switch | informal | long
    gen_seconds     REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tickets_lang ON tickets(lang);

-- Classifier tier: one row per batch run (a question set applied to the whole dataset).
CREATE TABLE IF NOT EXISTS classifier_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL DEFAULT (datetime('now')),
    question_set  TEXT NOT NULL,
    device        TEXT NOT NULL,
    model_version TEXT,
    n_tickets     INTEGER,
    seconds       REAL
);

-- The classifier's answers per ticket. *_conf = probability of the chosen answer;
-- *_probs = full distribution as JSON; churn_p / refund_p = P(yes).
CREATE TABLE IF NOT EXISTS classifier_predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL REFERENCES classifier_runs(id),
    ticket_id        INTEGER NOT NULL REFERENCES tickets(id),
    checkpoint       TEXT,          -- english | multilingual
    detected_lang    TEXT,          -- router detection (script-mapped for non-Latin)
    detected_script  TEXT,
    routing_reason   TEXT,
    sentiment        TEXT,  sentiment_conf  REAL, sentiment_probs  TEXT,
    department       TEXT,  department_conf REAL, department_probs TEXT,
    urgency          INTEGER, urgency_score REAL, urgency_conf REAL, urgency_probs TEXT,
    churn_p          REAL,
    refund_p         REAL,
    latency_ms       REAL,
    raw_json         TEXT,
    UNIQUE (run_id, ticket_id)
);

-- LLM tier labelling every ticket blind: the "LLM only" baseline, and a cache the
-- cascade reuses so policy sweeps need no new LLM calls.
CREATE TABLE IF NOT EXISTS llm_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    model       TEXT NOT NULL,
    n_tickets   INTEGER,
    seconds     REAL,
    prompt_version TEXT
);

CREATE TABLE IF NOT EXISTS llm_labels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES llm_runs(id),
    ticket_id       INTEGER NOT NULL REFERENCES tickets(id),
    sentiment       TEXT,
    department      TEXT,
    urgency         INTEGER,
    churn_risk      INTEGER,
    refund_request  INTEGER,
    summary         TEXT,
    seconds         REAL,
    output_tokens   INTEGER,
    ok              INTEGER NOT NULL DEFAULT 1,
    raw             TEXT,
    UNIQUE (run_id, ticket_id)
);

-- Cascade (LangGraph): final answer per ticket and why it was escalated.
CREATE TABLE IF NOT EXISTS cascade_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL DEFAULT (datetime('now')),
    policy      TEXT NOT NULL,      -- JSON
    n_tickets   INTEGER,
    escalated   INTEGER,
    seconds     REAL
);

CREATE TABLE IF NOT EXISTS cascade_results (
    run_id          INTEGER NOT NULL REFERENCES cascade_runs(id),
    ticket_id       INTEGER NOT NULL REFERENCES tickets(id),
    escalated       INTEGER NOT NULL,
    reasons         TEXT,           -- JSON list, e.g. ["low_confidence", "churn_screen"]
    source          TEXT NOT NULL,  -- classifier | llm
    sentiment       TEXT,
    department      TEXT,
    urgency         INTEGER,
    churn_risk      INTEGER,
    refund_request  INTEGER,
    summary         TEXT,
    cost_seconds    REAL,           -- classifier latency + LLM time if escalated
    PRIMARY KEY (run_id, ticket_id)
);
"""


def connect(path: Path | str = DB_PATH, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open (and create if needed) the database. Pass check_same_thread=False when a
    connection is shared across threads, as in the API's live triage endpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn
