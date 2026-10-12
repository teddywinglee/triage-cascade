"""
FastAPI backend for the dashboard.

Serves results from data/triage.sqlite, or from the committed snapshot in
sample_data/ when you haven't run the pipeline yourself. Policy evaluation reuses
pipeline/policy.py, so the dashboard and the CLI sweep always agree.

Run (project root, venv active):
    uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from statistics import median

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.config import DB_PATH, SAMPLE_DB_PATH
from pipeline.db import connect
from pipeline.evaluate import build_report
from pipeline.graph import latest_classifier_run, latest_llm_run
from pipeline.policy import (
    CLASSIFIER_ONLY,
    FIELDS,
    LLM_ONLY,
    Policy,
    escalation_reasons,
    evaluate_policy,
    final_answer,
)

app = FastAPI(title="Triage Cascade API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

# Display names of the models behind each tier, for the dashboard.
CLASSIFIER_NAME = "Laya"
LLM_NAME = "Gemma"


def db_path() -> Path:
    """Your own pipeline output if it exists, otherwise the committed sample data."""
    return DB_PATH if DB_PATH.exists() else SAMPLE_DB_PATH


# --------------------------------------------------------------------------- #
# Data loading (cached; POST /api/reload after re-running the pipeline)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def dataset() -> dict:
    path = db_path()
    conn = connect(path)
    classifier_run = latest_classifier_run(conn)
    llm_run = latest_llm_run(conn)
    if not classifier_run:
        raise HTTPException(503, "No classifier run yet. Run: python -m pipeline.classifier")
    rows = [dict(r) for r in conn.execute(
        """SELECT p.*, t.text, t.lang, t.channel, t.difficulty,
                  t.sentiment t_sentiment, t.department t_department, t.urgency t_urgency,
                  t.churn_risk t_churn, t.refund_request t_refund
           FROM classifier_predictions p JOIN tickets t ON t.id = p.ticket_id
           WHERE p.run_id = ? ORDER BY p.ticket_id""", (classifier_run,))]
    llm, llm_meta = {}, None
    if llm_run:
        llm = {r["ticket_id"]: dict(r) for r in conn.execute(
            "SELECT * FROM llm_labels WHERE run_id = ?", (llm_run,))}
        llm_meta = dict(conn.execute("SELECT * FROM llm_runs WHERE id = ?", (llm_run,)).fetchone())
    run = conn.execute("SELECT * FROM classifier_runs WHERE id = ?", (classifier_run,)).fetchone()
    report = build_report(conn, run)
    conn.close()
    return {"rows": rows, "llm": llm, "classifier_meta": dict(run), "llm_meta": llm_meta, "report": report,
            "source": "sample" if path == SAMPLE_DB_PATH else "local"}


class PolicyIn(BaseModel):
    confidence: float = Field(0.9, ge=0, le=1.01)
    churn_screen: float | None = 0.005
    churn_screen_neutral: bool = True
    refund_check: float | None = 0.5

    def to_policy(self, name: str = "custom") -> Policy:
        return Policy(name, confidence=self.confidence, churn_screen=self.churn_screen,
                      churn_screen_neutral=self.churn_screen_neutral, refund_check=self.refund_check)


def ok_llm(d: dict) -> dict:
    return {k: v for k, v in d["llm"].items() if v.get("ok")}


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/summary")
def summary():
    d = dataset()
    rows, llm = d["rows"], ok_llm(d)
    lat = sorted(r["latency_ms"] for r in rows)
    llm_secs = sorted(v["seconds"] for v in llm.values())
    by_lang: dict[str, int] = {}
    for r in rows:
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + 1
    return {
        "tickets": len(rows),
        "languages": by_lang,
        "data_source": d["source"],
        "classifier": {"name": CLASSIFIER_NAME, "run": d["classifier_meta"],
                       "p50_ms": lat[len(lat) // 2], "p95_ms": lat[int(len(lat) * 0.95)]},
        "llm": None if not d["llm_meta"] else {
            "name": LLM_NAME, "run": d["llm_meta"], "labelled": len(llm),
            "median_s": median(llm_secs) if llm_secs else None,
        },
    }


@app.post("/api/policy")
def policy(p: PolicyIn):
    """Metrics for one policy, both baselines, and the curve across confidence thresholds."""
    d = dataset()
    rows, llm = d["rows"], ok_llm(d)
    if not llm:
        raise HTTPException(503, "No LLM labels yet. Run: python -m pipeline.llm")
    curve = []
    for i in range(0, 101, 5):
        q = p.model_copy(update={"confidence": i / 100})
        m = evaluate_policy(rows, llm, q.to_policy())
        m["confidence"] = i / 100
        curve.append(m)
    return {
        "current": evaluate_policy(rows, llm, p.to_policy()),
        "classifier_only": evaluate_policy(rows, llm, CLASSIFIER_ONLY),
        "llm_only": evaluate_policy(rows, llm, LLM_ONLY),
        "curve": curve,
    }


@app.get("/api/calibration")
def calibration():
    rep = dataset()["report"]
    return {q: {"bins": rep[q]["calibration"]["bins"], "ece": rep[q]["calibration"]["ece"],
                "accuracy": rep[q]["accuracy"]} for q in ("sentiment", "department")}


@app.get("/api/languages")
def languages():
    """Per-language accuracy, classifier vs. LLM, for sentiment and department."""
    d = dataset()
    rows, llm = d["rows"], ok_llm(d)
    out = []
    for lang in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lang]
        g = [(r, llm[r["ticket_id"]]) for r in sub if r["ticket_id"] in llm]
        item = {"lang": lang, "n": len(sub)}
        for q in ("sentiment", "department"):
            item[f"classifier_{q}"] = round(sum(r[q] == r[f"t_{q}"] for r in sub) / len(sub), 4)
            item[f"llm_{q}"] = round(sum(x[q] == r[f"t_{q}"] for r, x in g) / len(g), 4) if g else None
        out.append(item)
    return out


def _ticket_view(r: dict, g: dict | None, p: Policy) -> dict:
    reasons = escalation_reasons(r, p)
    f = final_answer(r, g, reasons, p)
    truth = {"sentiment": r["t_sentiment"], "department": r["t_department"], "urgency": r["t_urgency"],
             "churn_risk": r["t_churn"], "refund_request": r["t_refund"]}
    return {
        "id": r["ticket_id"], "lang": r["lang"], "channel": r["channel"], "difficulty": r["difficulty"],
        "text": r["text"], "escalated": bool(reasons), "reasons": reasons, "source": f["source"],
        "truth": truth,
        "final": {k: f[k] for k in FIELDS},
        "correct": {k: f[k] == truth[k] for k in FIELDS},
    }


@app.post("/api/tickets")
def tickets(p: PolicyIn,
            lang: str | None = None,
            escalated: bool | None = None,
            wrong: str | None = Query(None, description="only tickets where this field is wrong"),
            difficulty: str | None = None):
    d = dataset()
    pol = p.to_policy()
    out = []
    for r in d["rows"]:
        if lang and r["lang"] != lang:
            continue
        if difficulty and r["difficulty"] != difficulty:
            continue
        v = _ticket_view(r, d["llm"].get(r["ticket_id"]), pol)
        if escalated is not None and v["escalated"] != escalated:
            continue
        if wrong and v["correct"].get(wrong, True):
            continue
        out.append(v)
    return out


@app.post("/api/tickets/{ticket_id}")
def ticket(ticket_id: int, p: PolicyIn):
    d = dataset()
    r = next((x for x in d["rows"] if x["ticket_id"] == ticket_id), None)
    if r is None:
        raise HTTPException(404, f"Ticket {ticket_id} not found")
    g = d["llm"].get(ticket_id)
    v = _ticket_view(r, g, p.to_policy())
    v["classifier"] = {
        "checkpoint": r["checkpoint"], "detected_lang": r["detected_lang"], "latency_ms": r["latency_ms"],
        "sentiment": r["sentiment"], "sentiment_conf": r["sentiment_conf"],
        "sentiment_probs": json.loads(r["sentiment_probs"] or "{}"),
        "department": r["department"], "department_conf": r["department_conf"],
        "department_probs": json.loads(r["department_probs"] or "{}"),
        "urgency_score": r["urgency_score"], "churn_p": r["churn_p"], "refund_p": r["refund_p"],
    }
    v["llm"] = None if not g else {k: g[k] for k in (*FIELDS, "summary", "seconds", "ok")}
    return v


class TriageIn(BaseModel):
    text: str = Field(min_length=3, max_length=4000)
    policy: PolicyIn = PolicyIn()


_live: dict = {}  # the Laya router and LLM client stay loaded between live requests


@app.post("/api/triage")
def triage(body: TriageIn):
    """Run one new message through the LangGraph cascade live (needs Laya installed and an LLM server)."""
    t0 = time.perf_counter()
    try:
        from pipeline.graph import Resources, build_graph

        res = _live.get("res")
        if res is None:
            # FastAPI runs sync endpoints in a thread pool, so the connection is shared across threads.
            res = _live["res"] = Resources(connect(db_path(), check_same_thread=False),
                                           body.policy.to_policy(), use_cache=False)
        res.policy = body.policy.to_policy()  # nodes read the policy at call time
        out = build_graph(res).invoke({"ticket_id": None, "text": body.text})
    except Exception as e:  # noqa: BLE001 - surface any model/runtime failure to the UI
        raise HTTPException(503, f"Live triage failed: {type(e).__name__}: {e}") from e
    pred = out["pred"]
    return {
        "seconds": round(time.perf_counter() - t0, 3),
        "classifier": {k: pred[k] for k in ("checkpoint", "detected_lang", "latency_ms", "sentiment",
                                             "sentiment_conf", "department", "department_conf",
                                             "urgency_score", "churn_p", "refund_p")},
        "reasons": out["reasons"],
        "llm": out.get("llm_labels"),
        "final": out["final"],
    }


@app.post("/api/reload")
def reload():
    dataset.cache_clear()
    return {"ok": True, "tickets": len(dataset()["rows"])}
