"""
Classifier tier: runs Laya over every ticket and stores its answers and confidences.

Laya is a non-generative decision model: an encoder scores each answer option of
each question in a single forward pass and returns a probability distribution.
It never writes text, so there is nothing to parse.

Each question set (see questions.py) becomes its own run, so wording changes can
be compared on the same tickets.

Usage (project root, venv active):
    python -m pipeline.classifier                    # run question sets v1 and v2
    python -m pipeline.classifier --sets v2          # just one set
    python -m pipeline.classifier --device cpu
    python -m pipeline.classifier --summary          # accuracy of the latest run per set, no model needed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version

from .db import connect
from .questions import QUESTION_SETS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Laya's router reports the script but not the language for non-Latin text.
SCRIPT_TO_LANG = {"han": "zh", "kana": "ja", "hangul": "ko", "devanagari": "hi",
                  "cyrillic": "ru", "arabic": "ar", "thai": "th", "hebrew": "he"}


def detection(result: dict) -> tuple[str | None, str | None, str | None]:
    routing = result.get("routing") or {}
    det = routing.get("detection") or {}
    script = det.get("script")
    lang = det.get("language") or SCRIPT_TO_LANG.get(script)
    return lang, script, routing.get("reason")


def flatten(ticket_id: int, result: dict, latency_ms: float) -> dict:
    """Turn Laya's nested answer dict into one classifier_predictions row."""
    a = result.get("answers", {})
    lang, script, reason = detection(result)
    s, d, u = a.get("sentiment", {}), a.get("department", {}), a.get("urgency", {})
    u_probs = u.get("probabilities") or {}
    return {
        "ticket_id": ticket_id,
        "checkpoint": (result.get("routing") or {}).get("model"),
        "detected_lang": lang,
        "detected_script": script,
        "routing_reason": reason,
        "sentiment": s.get("choice"),
        "sentiment_conf": s.get("answer_confidence"),
        "sentiment_probs": json.dumps(s.get("probabilities")),
        "department": d.get("choice"),
        "department_conf": d.get("answer_confidence"),
        "department_probs": json.dumps(d.get("probabilities")),
        # Most likely urgency class; the expected score is kept separately.
        "urgency": int(max(u_probs, key=u_probs.get)) if u_probs else None,
        "urgency_score": u.get("score"),
        "urgency_conf": u.get("answer_confidence"),
        "urgency_probs": json.dumps(u_probs),
        "churn_p": a.get("churn_risk", {}).get("noul"),
        "refund_p": a.get("refund_request", {}).get("noul"),
        "latency_ms": round(latency_ms, 2),
        "raw_json": json.dumps(result, ensure_ascii=False, default=str),
    }


def load_router(device: str):
    from laya import Router

    t0 = time.perf_counter()
    # Only the two checkpoints the pipeline needs (saves ~1.5 GB VRAM vs. all three).
    router = Router(device=device, max_loaded=2)
    router.preload(["english", "multilingual"])
    print(f"Laya loaded on {device} in {time.perf_counter() - t0:.1f}s")
    return router


def run_set(conn, router, set_name: str, device: str) -> int:
    questions = QUESTION_SETS[set_name]
    tickets = conn.execute("SELECT id, text FROM tickets ORDER BY id").fetchall()
    try:
        model_version = version("laya")
    except PackageNotFoundError:
        model_version = None

    run_id = conn.execute(
        "INSERT INTO classifier_runs (question_set, device, model_version, n_tickets) VALUES (?,?,?,?)",
        (set_name, device, model_version, len(tickets)),
    ).lastrowid

    router.predict(tickets[0]["text"], questions)  # warm-up
    sync = _cuda_sync(device)

    t0 = time.perf_counter()
    rows = []
    for i, t in enumerate(tickets, 1):
        s = time.perf_counter()
        result = router.predict(t["text"], questions)
        sync()
        rows.append(flatten(t["id"], result, (time.perf_counter() - s) * 1000))
        if i % 50 == 0 or i == len(tickets):
            print(f"  [{set_name}] {i}/{len(tickets)}")
    secs = time.perf_counter() - t0

    cols = list(rows[0].keys())
    conn.executemany(
        f"INSERT INTO classifier_predictions (run_id, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
        [(run_id, *[r[c] for c in cols]) for r in rows],
    )
    conn.execute("UPDATE classifier_runs SET seconds = ? WHERE id = ?", (round(secs, 2), run_id))
    conn.commit()
    print(f"  [{set_name}] run {run_id}: {len(rows)} tickets in {secs:.1f}s "
          f"({secs / len(rows) * 1000:.1f} ms/ticket)")
    return run_id


def _cuda_sync(device: str):
    if device != "cuda":
        return lambda: None
    import torch

    return torch.cuda.synchronize


SUMMARY_SQL = """
SELECT p.*, t.lang, t.sentiment AS t_sentiment, t.department AS t_department,
       t.urgency AS t_urgency, t.churn_risk AS t_churn, t.refund_request AS t_refund
FROM classifier_predictions p JOIN tickets t ON t.id = p.ticket_id
WHERE p.run_id = ?
"""


def summary(conn) -> None:
    runs = conn.execute(
        """SELECT r.* FROM classifier_runs r
           JOIN (SELECT question_set, MAX(id) AS id FROM classifier_runs GROUP BY question_set) m
             ON m.id = r.id ORDER BY r.question_set"""
    ).fetchall()
    if not runs:
        print("No classifier runs yet.")
        return

    print(f"\n{'accuracy':<16}" + "".join(f"{r['question_set'] + ' (run ' + str(r['id']) + ')':>16}" for r in runs))
    per_run = {}
    for r in runs:
        rows = conn.execute(SUMMARY_SQL, (r["id"],)).fetchall()
        n = len(rows)
        acc = lambda f, rows=rows, n=n: sum(f(x) for x in rows) / n  # noqa: E731
        per_run[r["id"]] = {
            "sentiment": acc(lambda x: x["sentiment"] == x["t_sentiment"]),
            "department": acc(lambda x: x["department"] == x["t_department"]),
            "urgency (exact)": acc(lambda x: x["urgency"] == x["t_urgency"]),
            "urgency (±1)": acc(lambda x: x["urgency"] is not None and abs(x["urgency"] - x["t_urgency"]) <= 1),
            "churn_risk": acc(lambda x: (x["churn_p"] > 0.5) == bool(x["t_churn"])),
            "refund_request": acc(lambda x: (x["refund_p"] > 0.5) == bool(x["t_refund"])),
            "language": acc(lambda x: x["detected_lang"] == x["lang"]),
            "ms/ticket": sum(x["latency_ms"] for x in rows) / n,
        }
    for metric in next(iter(per_run.values())):
        cells = []
        for r in runs:
            v = per_run[r["id"]][metric]
            cells.append(f"{v:>16.1f}" if metric == "ms/ticket" else f"{v:>16.1%}")
        print(f"{metric:<16}" + "".join(cells))

    # Department confusion for the newest run: department is the question whose wording differs.
    last = runs[-1]
    print(f"\nDepartment confusion, set {last['question_set']} (rows = truth, cols = Laya):")
    depts = ["billing", "technical", "account", "sales", "other"]
    grid = {(a, b): 0 for a in depts for b in depts}
    for x in conn.execute(SUMMARY_SQL, (last["id"],)).fetchall():
        if x["department"] in depts:
            grid[(x["t_department"], x["department"])] += 1
    print(f"{'':<11}" + "".join(f"{d:>10}" for d in depts))
    for a in depts:
        print(f"{a:<11}" + "".join(f"{grid[(a, b)]:>10}" for b in depts))


def main() -> None:
    p = argparse.ArgumentParser(description="Run the classifier tier (Laya) over all tickets")
    p.add_argument("--sets", nargs="+", default=list(QUESTION_SETS), choices=list(QUESTION_SETS))
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--summary", action="store_true", help="only print the summary of existing runs")
    args = p.parse_args()

    conn = connect()
    if not args.summary:
        n = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        if n == 0:
            sys.exit("No tickets yet. Run: python -m pipeline.generate --n 450")
        device = args.device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        router = load_router(device)
        for set_name in args.sets:
            run_set(conn, router, set_name, device)
    summary(conn)


if __name__ == "__main__":
    main()
