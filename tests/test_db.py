import sqlite3

import pytest

from pipeline.db import connect

TABLES = {
    "generation_runs", "tickets", "classifier_runs", "classifier_predictions",
    "llm_runs", "llm_labels", "cascade_runs", "cascade_results",
}


def table_names(conn):
    return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_connect_creates_schema_and_parent_dir(tmp_path):
    path = tmp_path / "nested" / "t.sqlite"
    conn = connect(path)
    assert path.exists()
    assert TABLES <= table_names(conn)


def test_connect_is_idempotent(tmp_path):
    path = tmp_path / "t.sqlite"
    connect(path).close()
    assert TABLES <= table_names(connect(path))


def test_rows_are_accessible_by_column_name(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    conn.execute("INSERT INTO generation_runs (model, requested) VALUES ('m', 3)")
    assert conn.execute("SELECT model, requested FROM generation_runs").fetchone()["requested"] == 3


def test_foreign_keys_are_enforced(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO classifier_predictions (run_id, ticket_id) VALUES (999, 999)")


def test_duplicate_ticket_text_is_rejected(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    row = ("hello", "chat", "en", "neutral", "other", 0, 0, 0, "plain")
    sql = """INSERT INTO tickets (text, channel, lang, sentiment, department, urgency,
             churn_risk, refund_request, difficulty) VALUES (?,?,?,?,?,?,?,?,?)"""
    conn.execute(sql, row)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, row)


