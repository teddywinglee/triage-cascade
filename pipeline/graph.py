"""
The cascade as a LangGraph flow, plus an offline policy sweep.

    START -> classifier (Laya) -> triage --(no reason)--> finalize -> END
                                     \\--(reasons)--> llm (Gemma) --/

Classifier answers come from the stored classifier run when available (live Laya
otherwise); LLM answers come from the stored LLM run when available (live LLM
otherwise). So a full routed run is fast and reproducible, while --text runs
everything live.

Usage:
    python -m pipeline.graph                         # route all tickets with the default policy
    python -m pipeline.graph --confidence 0.8 --no-refund-check
    python -m pipeline.graph --sweep                 # compare many policies (no model calls)
    python -m pipeline.graph --text "Mi app no sincroniza y tengo una reunión en 10 minutos"
    python -m pipeline.graph --mermaid               # print the graph diagram
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import DATA_DIR
from .db import connect
from .policy import CLASSIFIER_ONLY, LLM_ONLY, Policy, escalation_reasons, evaluate_policy, final_answer
from .questions import QUESTION_SETS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SWEEP_PATH = DATA_DIR / "policy_sweep.json"


class TicketState(TypedDict, total=False):
    ticket_id: int | None
    text: str
    pred: dict          # classifier prediction (flattened, as stored in `classifier_predictions`)
    reasons: list[str]  # why it was escalated; empty = the classifier's answer stands
    llm_labels: dict | None  # LLM labels (None when not escalated)
    final: dict
    timings: dict


def latest_classifier_run(conn, question_set: str = "v1") -> int | None:
    return conn.execute("SELECT MAX(id) FROM classifier_runs WHERE question_set = ?", (question_set,)).fetchone()[0]


def latest_llm_run(conn) -> int | None:
    return conn.execute("SELECT MAX(id) FROM llm_runs").fetchone()[0]


# --------------------------------------------------------------------------- #
# Lazy resources (only loaded when a node needs them live)
# --------------------------------------------------------------------------- #
class Resources:
    def __init__(self, conn, policy: Policy, use_cache: bool = True):
        self.conn, self.policy, self.use_cache = conn, policy, use_cache
        self._router = self._llm = None
        self.classifier_run = latest_classifier_run(conn)
        self.llm_run = latest_llm_run(conn)

    def router(self):
        if self._router is None:
            import torch

            from .classifier import load_router
            self._router = load_router("cuda" if torch.cuda.is_available() else "cpu")
        return self._router

    def llm(self):
        if self._llm is None:
            from .llm import make_llm
            self._llm = make_llm()
        return self._llm


def build_graph(res: Resources):
    def classifier(state: TicketState) -> TicketState:
        t0 = time.perf_counter()
        row = None
        if state.get("ticket_id") and res.use_cache and res.classifier_run:
            row = res.conn.execute("SELECT * FROM classifier_predictions WHERE run_id = ? AND ticket_id = ?",
                                   (res.classifier_run, state["ticket_id"])).fetchone()
        if row:
            pred = dict(row)
        else:
            from .classifier import flatten
            s = time.perf_counter()
            result = res.router().predict(state["text"], QUESTION_SETS["v1"])
            pred = flatten(state.get("ticket_id") or 0, result, (time.perf_counter() - s) * 1000)
        return {"pred": pred, "timings": {"classifier_s": round(time.perf_counter() - t0, 4)}}

    def triage(state: TicketState) -> TicketState:
        return {"reasons": escalation_reasons(state["pred"], res.policy)}

    def llm(state: TicketState) -> TicketState:
        t0 = time.perf_counter()
        row = None
        if state.get("ticket_id") and res.use_cache and res.llm_run:
            row = res.conn.execute(
                # Failed cached labels (ok = 0) are reused too: finalize falls back to the classifier.
                "SELECT * FROM llm_labels WHERE run_id = ? AND ticket_id = ?",
                (res.llm_run, state["ticket_id"])).fetchone()
        if row:
            labels = dict(row)
        else:
            from .llm import label_ticket
            lab, secs, _, _ = label_ticket(res.llm(), state["text"])
            labels = ({**lab.model_dump(), "churn_risk": int(lab.churn_risk),
                       "refund_request": int(lab.refund_request), "seconds": secs, "ok": 1}
                      if lab else {"ok": 0, "seconds": secs})
        timings = {**state.get("timings", {}), "llm_s": round(time.perf_counter() - t0, 4)}
        return {"llm_labels": labels, "timings": timings}

    def finalize(state: TicketState) -> TicketState:
        return {"final": final_answer(state["pred"], state.get("llm_labels"), state["reasons"], res.policy)}

    def route(state: TicketState) -> str:
        return "llm" if state["reasons"] else "finalize"

    g = StateGraph(TicketState)
    g.add_node("classifier", classifier)
    g.add_node("triage", triage)
    g.add_node("llm", llm)
    g.add_node("finalize", finalize)
    g.add_edge(START, "classifier")
    g.add_edge("classifier", "triage")
    g.add_conditional_edges("triage", route, {"llm": "llm", "finalize": "finalize"})
    g.add_edge("llm", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_all(conn, policy: Policy, use_cache: bool) -> None:
    res = Resources(conn, policy, use_cache)
    if use_cache and not res.llm_run:
        print("Note: no stored LLM labels yet (run `python -m pipeline.llm`); escalations will call the LLM live.")
    app = build_graph(res)
    tickets = conn.execute("SELECT id, text FROM tickets ORDER BY id").fetchall()
    run_id = conn.execute("INSERT INTO cascade_runs (policy, n_tickets) VALUES (?, ?)",
                          (json.dumps(policy.to_dict()), len(tickets))).lastrowid
    t0, n_esc = time.perf_counter(), 0
    for i, t in enumerate(tickets, 1):
        out = app.invoke({"ticket_id": t["id"], "text": t["text"]})
        f, pred, labels = out["final"], out["pred"], out.get("llm_labels")
        esc = bool(out["reasons"])
        n_esc += esc
        cost = pred["latency_ms"] / 1000 + (labels["seconds"] if esc and labels else 0.0)
        conn.execute(
            """INSERT INTO cascade_results (run_id, ticket_id, escalated, reasons, source, sentiment, department,
               urgency, churn_risk, refund_request, summary, cost_seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, t["id"], int(esc), json.dumps(out["reasons"]), f["source"], f["sentiment"], f["department"],
             f["urgency"], f["churn_risk"], f["refund_request"], f["summary"], round(cost, 4)))
        if i % 100 == 0:
            print(f"  {i}/{len(tickets)}")
    conn.execute("UPDATE cascade_runs SET escalated = ?, seconds = ? WHERE id = ?",
                 (n_esc, round(time.perf_counter() - t0, 2), run_id))
    conn.commit()
    print(f"Cascade run {run_id}: {len(tickets)} tickets, {n_esc} escalated ({n_esc / len(tickets):.1%}) "
          f"with policy {policy.to_dict()}")


def sweep_rows(conn, llm_run: int | None = None):
    classifier_run = latest_classifier_run(conn)
    llm_run = llm_run or latest_llm_run(conn)
    if not classifier_run or not llm_run:
        sys.exit("Need a classifier run (pipeline.classifier) and an LLM run (pipeline.llm) first.")
    rows = [dict(r) for r in conn.execute(
        """SELECT p.*, t.sentiment t_sentiment, t.department t_department, t.urgency t_urgency,
                  t.churn_risk t_churn, t.refund_request t_refund
           FROM classifier_predictions p JOIN tickets t ON t.id = p.ticket_id WHERE p.run_id = ?""",
        (classifier_run,))]
    llm = {r["ticket_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM llm_labels WHERE run_id = ? AND ok = 1", (llm_run,))}
    return rows, llm, llm_run


def sweep(conn, llm_run: int | None = None) -> None:
    rows, llm, llm_run = sweep_rows(conn, llm_run)
    pv = conn.execute("SELECT prompt_version FROM llm_runs WHERE id = ?", (llm_run,)).fetchone()[0]
    print(f"Using LLM run {llm_run} (prompt: {pv})")
    policies = [CLASSIFIER_ONLY, LLM_ONLY]
    for conf in (0.0, 0.7, 0.8, 0.85, 0.9, 0.95):
        for churn in (None, "neg", "neg+neu"):
            for refund in (None, 0.5):
                if conf == 0.0 and churn is None and refund is None:
                    continue  # same as classifier_only
                policies.append(Policy(
                    f"c{conf}_ch-{churn or 'off'}_rf{refund or 'off'}", confidence=conf,
                    churn_screen=0.005 if churn else None, churn_screen_neutral=churn == "neg+neu",
                    refund_check=refund))
    results = [evaluate_policy(rows, llm, p) for p in policies]
    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_PATH.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"\n{'policy':<30}{'escal.':>7}{'sent':>7}{'dept':>7}{'urg':>7}{'churn P/R':>13}"
          f"{'refund P/R':>13}{'s/ticket':>10}{'total min':>10}")
    for r in sorted(results, key=lambda r: r["escalation_rate"]):
        print(f"{r['policy']['name']:<30}{r['escalation_rate']:>7.0%}{r['sentiment']:>7.0%}{r['department']:>7.0%}"
              f"{r['urgency']:>7.0%}{r['churn']['precision']:>7.0%}/{r['churn']['recall']:<5.0%}"
              f"{r['refund']['precision']:>7.0%}/{r['refund']['recall']:<5.0%}"
              f"{r['seconds_per_ticket']:>10.2f}{r['total_minutes']:>10.1f}")
    print("\nPolicy names: c<confidence>_ch-<churn screen on: off | neg | neg+neu>_rf<refund check>")
    print(f"Full results: {SWEEP_PATH}")


def run_text(conn, policy: Policy, text: str) -> None:
    app = build_graph(Resources(conn, policy, use_cache=False))
    out = app.invoke({"ticket_id": None, "text": text})
    pred = out["pred"]
    print(f"Classifier ({pred['latency_ms']:.0f} ms, {pred['checkpoint']}, lang={pred['detected_lang']}): "
          f"{pred['sentiment']} ({pred['sentiment_conf']:.2f}), {pred['department']} ({pred['department_conf']:.2f}), "
          f"churn_p={pred['churn_p']:.3f}, refund_p={pred['refund_p']:.3f}")
    print(f"Escalation reasons: {out['reasons'] or 'none, the classifier answer stands'}")
    if out.get("llm_labels"):
        labels = out["llm_labels"]
        print(f"LLM ({labels.get('seconds', 0):.1f}s): {json.dumps(labels, ensure_ascii=False)}")
    print(f"Final: {json.dumps(out['final'], ensure_ascii=False)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Classifier-first triage cascade (LangGraph)")
    ap.add_argument("--sweep", action="store_true", help="compare policies offline")
    ap.add_argument("--llm-run", type=int, default=None, help="with --sweep: which LLM run (default latest)")
    ap.add_argument("--text", help="run one ad-hoc message through the graph, fully live")
    ap.add_argument("--mermaid", action="store_true", help="print the graph as a Mermaid diagram")
    ap.add_argument("--no-cache", action="store_true", help="call both models live instead of using stored answers")
    ap.add_argument("--confidence", type=float, default=Policy.confidence)
    ap.add_argument("--churn-screen", type=float, default=Policy.churn_screen)
    ap.add_argument("--no-churn-screen", action="store_true")
    ap.add_argument("--refund-check", type=float, default=Policy.refund_check)
    ap.add_argument("--no-refund-check", action="store_true")
    args = ap.parse_args()

    conn = connect()
    policy = Policy("cli", confidence=args.confidence,
                    churn_screen=None if args.no_churn_screen else args.churn_screen,
                    refund_check=None if args.no_refund_check else args.refund_check)
    if args.mermaid:
        print(build_graph(Resources(conn, policy)).get_graph().draw_mermaid())
    elif args.sweep:
        sweep(conn, args.llm_run)
    elif args.text:
        run_text(conn, policy, args.text)
    else:
        run_all(conn, policy, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
