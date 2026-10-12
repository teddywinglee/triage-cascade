"""API smoke tests against the committed sample data (no GPU or LLM needed)."""

import pytest
from fastapi.testclient import TestClient

import api.main as main


@pytest.fixture(scope="module")
def client():
    main.db_path = lambda: main.SAMPLE_DB_PATH  # ignore any local data/ database
    main.dataset.cache_clear()
    return TestClient(main.app)


def test_summary(client):
    s = client.get("/api/summary").json()
    assert s["tickets"] == 450 and len(s["languages"]) == 9
    assert s["classifier"]["name"] == "Laya" and s["llm"]["labelled"] == 450


def test_policy_reproduces_readme_numbers(client):
    r = client.post("/api/policy", json={"confidence": 0.9}).json()
    assert r["current"]["escalation_rate"] == pytest.approx(0.70, abs=0.01)
    assert r["current"]["department"] == pytest.approx(0.91, abs=0.01)
    assert r["classifier_only"]["department"] == pytest.approx(0.70, abs=0.01)
    assert r["llm_only"]["department"] == pytest.approx(0.97, abs=0.01)
    assert len(r["curve"]) == 21


def test_ticket_detail(client):
    t = client.post("/api/tickets/13", json={}).json()
    assert t["id"] == 13 and t["classifier"]["sentiment_probs"]
    assert client.post("/api/tickets/99999", json={}).status_code == 404


def test_ticket_filters(client):
    sarcasm = client.post("/api/tickets?difficulty=sarcasm", json={}).json()
    assert len(sarcasm) == 9 and all(t["difficulty"] == "sarcasm" for t in sarcasm)
