"""
Escalation policy: pure functions shared by the LangGraph flow, the policy sweep and the API.

Rules (each one follows from the classifier evaluation, see evaluate.py):
  - low_confidence : the classifier's sentiment or department confidence is below `confidence`
                     (confidence ranks right vs. wrong answers well, even though it's overstated)
  - churn_screen   : the classifier's churn probability is above a very low cut-off on a
                     negative or neutral ticket (its churn answer at 0.5 misses ~97% of threats,
                     but the score still works as a cheap filter)
  - refund_check   : the classifier thinks it's a refund request (high recall, ~58% precision)

Tickets with no reason keep the classifier's answers, with urgency taken from cut-offs on
its urgency score (its own urgency class does worse than always guessing "not urgent").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

FIELDS = ("sentiment", "department", "urgency", "churn_risk", "refund_request")


@dataclass
class Policy:
    name: str = "default"
    confidence: float = 0.9               # escalate if min(sentiment, department) conf < this; >1 = always
    churn_screen: float | None = 0.005    # escalate tickets with churn_p above this; None = off
    churn_screen_neutral: bool = True     # screen neutral tickets too, not only negative ones
    refund_check: float | None = 0.5      # escalate if refund_p above this; None = off
    churn_keep: float = 0.5               # classifier's churn decision when not escalated
    refund_keep: float = 0.7              # classifier's refund decision when not escalated (cross-validated)
    urgency_t1: float = 1.6553            # urgency score cut-offs (cross-validated, see evaluate.py)
    urgency_t2: float = 1.841

    def to_dict(self) -> dict:
        return asdict(self)


CLASSIFIER_ONLY = Policy("classifier_only", confidence=0.0, churn_screen=None, refund_check=None)
LLM_ONLY = Policy("llm_only", confidence=1.01, churn_screen=None, refund_check=None)


def escalation_reasons(pred: dict, p: Policy) -> list[str]:
    """Why a ticket should go to the LLM, given the classifier's prediction. Empty = keep it."""
    reasons = []
    if p.confidence > 1 or min(pred["sentiment_conf"], pred["department_conf"]) < p.confidence:
        reasons.append("low_confidence" if p.confidence <= 1 else "always")
    screened = ("negative", "neutral") if p.churn_screen_neutral else ("negative",)
    if p.churn_screen is not None and pred["sentiment"] in screened and pred["churn_p"] > p.churn_screen:
        reasons.append("churn_screen")
    if p.refund_check is not None and pred["refund_p"] > p.refund_check:
        reasons.append("refund_check")
    return reasons


def classifier_answer(pred: dict, p: Policy) -> dict:
    score = pred["urgency_score"]
    return {
        "source": "classifier",
        "sentiment": pred["sentiment"],
        "department": pred["department"],
        "urgency": 0 if score < p.urgency_t1 else (1 if score < p.urgency_t2 else 2),
        "churn_risk": int(pred["churn_p"] > p.churn_keep),
        "refund_request": int(pred["refund_p"] > p.refund_keep),
        "summary": None,
    }


def final_answer(pred: dict, llm: dict | None, reasons: list[str], p: Policy) -> dict:
    """Escalated and the LLM answered -> the LLM's labels; otherwise the classifier's."""
    if reasons and llm and llm.get("ok", 1):
        return {"source": "llm", **{k: llm[k] for k in (*FIELDS, "summary")}}
    return classifier_answer(pred, p)


# --------------------------------------------------------------------------- #
# Sweep: evaluate many policies offline using cached classifier + LLM answers.
# --------------------------------------------------------------------------- #
def _prf(pairs):
    tp = sum(p and t for p, t in pairs)
    fp = sum(p and not t for p, t in pairs)
    fn = sum(not p and t for p, t in pairs)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return round(prec, 4), round(rec, 4), round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 4)


def evaluate_policy(rows: list[dict], llm: dict[int, dict], p: Policy) -> dict:
    """rows: classifier prediction + ground truth per ticket; llm: ticket_id -> LLM labels."""
    finals, n_esc, cost = [], 0, 0.0
    reason_counts: dict[str, int] = {}
    for r in rows:
        reasons = escalation_reasons(r, p)
        g = llm.get(r["ticket_id"])
        f = final_answer(r, g, reasons, p)
        if reasons:
            n_esc += 1
            for x in reasons:
                reason_counts[x] = reason_counts.get(x, 0) + 1
        classifier_s = 0.0 if p.confidence > 1 else r["latency_ms"] / 1000
        cost += classifier_s + (g["seconds"] if reasons and g else 0.0)
        finals.append((r, f))
    n = len(rows)
    acc = lambda k, tk: round(sum(f[k] == r[tk] for r, f in finals) / n, 4)  # noqa: E731
    cp, cr, cf = _prf([(bool(f["churn_risk"]), bool(r["t_churn"])) for r, f in finals])
    rp, rr, rf = _prf([(bool(f["refund_request"]), bool(r["t_refund"])) for r, f in finals])
    return {
        "policy": p.to_dict(),
        "n": n,
        "escalated": n_esc,
        "escalation_rate": round(n_esc / n, 4),
        "reasons": reason_counts,
        "sentiment": acc("sentiment", "t_sentiment"),
        "department": acc("department", "t_department"),
        "urgency": acc("urgency", "t_urgency"),
        "churn": {"precision": cp, "recall": cr, "f1": cf},
        "refund": {"precision": rp, "recall": rr, "f1": rf},
        "seconds_per_ticket": round(cost / n, 4),
        "total_minutes": round(cost / 60, 2),
    }
