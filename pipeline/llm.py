"""
LLM tier: a generative LLM (Gemma 4 12B via an OpenAI-compatible server) labels every
ticket blind.

Two purposes:
  1. Baseline: "what if every ticket went to the LLM?"
  2. Cache: the cascade (graph.py) reuses these answers, so trying different
     escalation policies needs no new LLM calls.

Caveat: the same model generated the tickets, so its scores here are likely
optimistic. It is labelling text it wrote itself.

Usage:
    python -m pipeline.llm --limit 10     # quick check
    python -m pipeline.llm                # all tickets (~10-15 min)
    python -m pipeline.llm --summary      # accuracy of the latest run, no LLM needed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from . import config
from .db import connect
from .questions import QUESTION_SETS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Labels(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    department: Literal["billing", "technical", "account", "sales", "other"]
    urgency: int = Field(ge=0, le=2)
    churn_risk: bool
    refund_request: bool
    summary: str


# The label definitions come from the SAME question set the classifier gets
# (questions.py, v1), not from generate.py. Reusing the generator's wording leaked its
# phrasing to the LLM (run 1, prompt "generator-wording": department 99% vs. 97%),
# which made the comparison with the classifier unfair.
PROMPT_VERSION = "v1-shared-with-classifier"
_Q = QUESTION_SETS["v1"]


def _options(q: dict) -> str:
    return "\n".join(f'    "{k}": {v}' for k, v in q["criteria"].items())


SYSTEM_PROMPT = f"""You triage customer support messages.

Read the customer's message (it may be in any language) and return JSON with exactly these keys:
- "sentiment": {_Q["sentiment"]["instructions"]} One of:
{_options(_Q["sentiment"])}
- "department": {_Q["department"]["instructions"]} One of:
{_options(_Q["department"])}
- "urgency": {_Q["urgency"]["instructions"]} """ + ", ".join(
    f"{i} = {c}" for i, c in enumerate(_Q["urgency"]["criteria"])) + f"""
- "churn_risk": true/false. {_Q["churn_risk"]["instructions"]}
- "refund_request": true/false. {_Q["refund_request"]["instructions"]}
- "summary": one short English sentence describing the issue

Respond with the JSON object only."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def make_llm(max_tokens: int = 400):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(base_url=config.LLM_BASE_URL, api_key="lm-studio", model=config.LLM_MODEL,
                      temperature=0.0, max_tokens=max_tokens, timeout=300)


def label_ticket(llm, text: str, retries: int = 2) -> tuple[Labels | None, float, int | None, str]:
    """Returns (labels or None, seconds, output tokens, raw text of the last attempt)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    msgs = [SystemMessage(SYSTEM_PROMPT), HumanMessage(f"Customer message:\n{text}")]
    start, raw, tokens = time.perf_counter(), "", None
    for _ in range(retries + 1):
        try:
            msg = llm.invoke(msgs)
        except Exception as e:  # noqa: BLE001
            raw = f"{type(e).__name__}: {e}"
            time.sleep(2)
            continue
        raw = msg.content or ""
        tokens = (getattr(msg, "usage_metadata", None) or {}).get("output_tokens")
        m = _JSON_RE.search(raw)
        if not m:
            continue
        try:
            return Labels.model_validate(json.loads(m.group(0), strict=False)), time.perf_counter() - start, tokens, raw
        except (ValueError, ValidationError):
            continue
    return None, time.perf_counter() - start, tokens, raw


def latest_run_id(conn) -> int | None:
    row = conn.execute("SELECT MAX(id) FROM llm_runs").fetchone()
    return row[0] if row else None


def summary(conn, run_id: int | None = None) -> None:
    run_id = run_id or latest_run_id(conn)
    if not run_id:
        print("No LLM runs yet.")
        return
    rows = conn.execute(
        """SELECT s.*, t.sentiment ts, t.department td, t.urgency tu, t.churn_risk tc, t.refund_request tr
           FROM llm_labels s JOIN tickets t ON t.id = s.ticket_id WHERE s.run_id = ?""", (run_id,)
    ).fetchall()
    ok = [r for r in rows if r["ok"]]
    if not ok:
        print("No successful labels in this run.")
        return
    n = len(ok)
    acc = lambda f: sum(f(r) for r in ok) / n  # noqa: E731

    def pr(pred, truth):
        tp = sum(r[pred] and r[truth] for r in ok)
        fp = sum(r[pred] and not r[truth] for r in ok)
        fn = sum(not r[pred] and r[truth] for r in ok)
        return (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)

    secs = sorted(r["seconds"] for r in ok)
    print(f"\nLLM run {run_id} ({config.LLM_MODEL}): {n}/{len(rows)} labelled, "
          f"median {secs[n // 2]:.2f}s/ticket, total {sum(secs) / 60:.1f} min")
    print(f"  sentiment   {acc(lambda r: r['sentiment'] == r['ts']):.1%}")
    print(f"  department  {acc(lambda r: r['department'] == r['td']):.1%}")
    print(f"  urgency     {acc(lambda r: r['urgency'] == r['tu']):.1%} exact")
    for name, p, t in (("churn", "churn_risk", "tc"), ("refund", "refund_request", "tr")):
        prec, rec = pr(p, t)
        print(f"  {name:<11} precision {prec:.1%}  recall {rec:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Label all tickets with the LLM tier")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tickets")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--run-id", type=int, default=None, help="with --summary: which run")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    if args.show_prompt:
        print(SYSTEM_PROMPT)
        return
    conn = connect()
    if args.summary:
        summary(conn, args.run_id)
        return

    tickets = conn.execute("SELECT id, text FROM tickets ORDER BY id").fetchall()
    if args.limit:
        tickets = tickets[: args.limit]
    llm = make_llm()
    run_id = conn.execute("INSERT INTO llm_runs (model, n_tickets, prompt_version) VALUES (?, ?, ?)",
                          (config.LLM_MODEL, len(tickets), PROMPT_VERSION)).lastrowid
    conn.commit()
    print(f"LLM run {run_id}: {len(tickets)} tickets with {config.LLM_MODEL} @ {config.LLM_BASE_URL} "
          f"(prompt {PROMPT_VERSION})")

    t0 = time.perf_counter()
    fails = 0
    for i, t in enumerate(tickets, 1):
        lab, secs, tokens, raw = label_ticket(llm, t["text"])
        if lab is None:
            fails += 1
        conn.execute(
            """INSERT INTO llm_labels (run_id, ticket_id, sentiment, department, urgency, churn_risk,
               refund_request, summary, seconds, output_tokens, ok, raw) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, t["id"],
             lab.sentiment if lab else None, lab.department if lab else None, lab.urgency if lab else None,
             int(lab.churn_risk) if lab else None, int(lab.refund_request) if lab else None,
             lab.summary if lab else None, round(secs, 3), tokens, int(lab is not None), raw),
        )
        conn.commit()
        if i % 10 == 0 or i == len(tickets) or lab is None:
            eta = (time.perf_counter() - t0) / i * (len(tickets) - i)
            status = f"{lab.sentiment}/{lab.department}/u{lab.urgency}" if lab else "FAILED"
            print(f"  [{i:>4}/{len(tickets)}] {secs:5.2f}s  {status:<28} ETA {eta / 60:4.1f}m")

    elapsed = time.perf_counter() - t0
    conn.execute("UPDATE llm_runs SET seconds = ? WHERE id = ?", (round(elapsed, 1), run_id))
    conn.commit()
    print(f"\nDone in {elapsed / 60:.1f} min, {fails} failed.")
    summary(conn, run_id)


if __name__ == "__main__":
    main()
