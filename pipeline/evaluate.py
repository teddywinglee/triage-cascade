"""
Evaluate the classifier tier (Laya) against the generated ground truth.

Standard library only. Writes data/eval_report.json (consumed by the dashboard)
and prints a readable summary.

What it measures
  - choice questions (sentiment, department): accuracy, per-class precision/recall/F1,
    confusion matrix, calibration (reliability bins + ECE), accuracy vs. confidence threshold
  - yes/no questions (churn, refund): AUC, precision/recall at 0.5, and a
    cross-validated "best F1" threshold
  - urgency: the classifier's own class vs. cut-offs on its score, chosen by cross-validation
  - breakdowns by language, difficulty style and checkpoint
  - escalation curve: share of tickets escalated vs. accuracy of the ones the classifier keeps

Usage:
    python -m pipeline.evaluate                  # latest run of question set v1
    python -m pipeline.evaluate --set v2
    python -m pipeline.evaluate --run-id 1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from statistics import median

from .config import DATA_DIR
from .db import connect

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPORT_PATH = DATA_DIR / "eval_report.json"
SENTIMENTS = ["positive", "neutral", "negative"]
DEPARTMENTS = ["billing", "technical", "account", "sales", "other"]
K_FOLDS = 5

ROWS_SQL = """
SELECT p.ticket_id, p.checkpoint, p.detected_lang, p.detected_script,
       p.sentiment, p.sentiment_conf, p.department, p.department_conf,
       p.urgency, p.urgency_score, p.urgency_conf, p.churn_p, p.refund_p, p.latency_ms,
       t.lang, t.channel, t.difficulty,
       t.sentiment AS t_sentiment, t.department AS t_department, t.urgency AS t_urgency,
       t.churn_risk AS t_churn, t.refund_request AS t_refund
FROM classifier_predictions p JOIN tickets t ON t.id = p.ticket_id
WHERE p.run_id = ?
ORDER BY p.ticket_id
"""


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def r4(x):
    return None if x is None else round(x, 4)


def choice_metrics(rows, pred, conf, truth, labels) -> dict:
    n = len(rows)
    correct = [r[pred] == r[truth] for r in rows]
    per_class = {}
    for c in labels:
        tp = sum(r[pred] == c and r[truth] == c for r in rows)
        fp = sum(r[pred] == c and r[truth] != c for r in rows)
        fn = sum(r[pred] != c and r[truth] == c for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        per_class[c] = {"precision": r4(p), "recall": r4(rc), "f1": r4(f1), "support": tp + fn}
    confusion = [[sum(r[truth] == a and r[pred] == b for r in rows) for b in labels] for a in labels]
    return {
        "n": n,
        "accuracy": r4(sum(correct) / n),
        "macro_f1": r4(sum(v["f1"] for v in per_class.values()) / len(labels)),
        "per_class": per_class,
        "labels": labels,
        "confusion": confusion,  # rows = truth, cols = predicted
        "calibration": calibration(rows, conf, correct),
        "selective": selective_curve([r[conf] for r in rows], correct),
    }


def calibration(rows, conf, correct, bins: int = 10) -> dict:
    """Reliability diagram data + expected calibration error (ECE)."""
    buckets = defaultdict(list)
    for r, ok in zip(rows, correct, strict=True):
        buckets[min(int(r[conf] * bins), bins - 1)].append((r[conf], ok))
    out, ece, n = [], 0.0, len(rows)
    for b in range(bins):
        items = buckets.get(b, [])
        if not items:
            continue
        mean_conf = sum(c for c, _ in items) / len(items)
        acc = sum(ok for _, ok in items) / len(items)
        ece += len(items) / n * abs(acc - mean_conf)
        out.append({"bin_low": b / bins, "bin_high": (b + 1) / bins, "n": len(items),
                    "mean_confidence": r4(mean_conf), "accuracy": r4(acc)})
    return {"bins": out, "ece": r4(ece)}


def selective_curve(confs, correct) -> list[dict]:
    """For each confidence threshold: how many tickets the classifier keeps and how accurate they are."""
    out = []
    for i in range(0, 100, 5):
        t = i / 100
        kept = [ok for c, ok in zip(confs, correct, strict=True) if c >= t]
        out.append({"threshold": t, "coverage": r4(len(kept) / len(confs)),
                    "accuracy": r4(sum(kept) / len(kept)) if kept else None})
    return out


def auc(scores, labels) -> float | None:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def prf(scores, labels, t) -> dict:
    tp = sum(s > t and y for s, y in zip(scores, labels, strict=True))
    fp = sum(s > t and not y for s, y in zip(scores, labels, strict=True))
    fn = sum(s <= t and y for s, y in zip(scores, labels, strict=True))
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": t, "precision": r4(p), "recall": r4(rc),
            "f1": r4(2 * p * rc / (p + rc) if p + rc else 0.0), "tp": tp, "fp": fp, "fn": fn}


def folds(rows):
    return [[r for r in rows if r["ticket_id"] % K_FOLDS == k] for k in range(K_FOLDS)]


def binary_metrics(rows, score_key, truth_key) -> dict:
    scores = [r[score_key] for r in rows]
    labels = [bool(r[truth_key]) for r in rows]
    grid = sorted({0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001})

    def best_t(sub):
        s = [r[score_key] for r in sub]
        y = [bool(r[truth_key]) for r in sub]
        return max(grid, key=lambda t: prf(s, y, t)["f1"])

    # Cross-validated: pick the threshold on 4/5 of tickets, score it on the held-out 1/5.
    tp = fp = fn = 0
    chosen = []
    for k, test in enumerate(folds(rows)):
        train = [r for r in rows if r["ticket_id"] % K_FOLDS != k]
        t = best_t(train)
        chosen.append(t)
        m = prf([r[score_key] for r in test], [bool(r[truth_key]) for r in test], t)
        tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(rows),
        "positives": sum(labels),
        "base_rate": r4(sum(labels) / len(labels)),
        "auc": r4(auc(scores, labels)),
        "accuracy_at_0_5": r4(sum((s > 0.5) == y for s, y in zip(scores, labels, strict=True)) / len(labels)),
        "at_0_5": prf(scores, labels, 0.5),
        "curve": [prf(scores, labels, t) for t in grid],
        "cv_best_f1": {"thresholds_chosen": chosen, "precision": r4(p), "recall": r4(rc),
                       "f1": r4(2 * p * rc / (p + rc) if p + rc else 0.0)},
    }


def urgency_metrics(rows) -> dict:
    def classify(score, t1, t2):
        return 0 if score < t1 else (1 if score < t2 else 2)

    def acc(sub, t1, t2):
        return sum(classify(r["urgency_score"], t1, t2) == r["t_urgency"] for r in sub) / len(sub)

    def best(sub):
        qs = sorted(r["urgency_score"] for r in sub)
        cand = sorted({round(qs[int(i / 40 * (len(qs) - 1))], 4) for i in range(41)})
        return max(((a, b) for a in cand for b in cand if a < b), key=lambda ab: acc(sub, *ab))

    held_out = []
    for k, test in enumerate(folds(rows)):
        t1, t2 = best([r for r in rows if r["ticket_id"] % K_FOLDS != k])
        held_out += [classify(r["urgency_score"], t1, t2) == r["t_urgency"] for r in test]
    t1, t2 = best(rows)
    by_true = {u: [r["urgency_score"] for r in rows if r["t_urgency"] == u] for u in (0, 1, 2)}
    return {
        "class_accuracy": r4(sum(r["urgency"] == r["t_urgency"] for r in rows) / len(rows)),
        "class_distribution": dict(Counter(str(r["urgency"]) for r in rows)),
        "score_median_by_true_class": {str(u): r4(median(v)) for u, v in by_true.items() if v},
        "blocking_auc": r4(auc([r["urgency_score"] for r in rows], [r["t_urgency"] == 2 for r in rows])),
        "tuned_cutoffs": {"t1": t1, "t2": t2, "in_sample_accuracy": r4(acc(rows, t1, t2)),
                          "cv_accuracy": r4(sum(held_out) / len(held_out))},
        "majority_baseline": r4(max(Counter(r["t_urgency"] for r in rows).values()) / len(rows)),
    }


def breakdown(rows, key) -> dict:
    out = {}
    for val in sorted({r[key] for r in rows}, key=str):
        sub = [r for r in rows if r[key] == val]
        out[str(val)] = {
            "n": len(sub),
            "sentiment": r4(sum(r["sentiment"] == r["t_sentiment"] for r in sub) / len(sub)),
            "department": r4(sum(r["department"] == r["t_department"] for r in sub) / len(sub)),
            "language_detected": r4(sum(r["detected_lang"] == r["lang"] for r in sub) / len(sub)),
            "mean_sentiment_conf": r4(sum(r["sentiment_conf"] for r in sub) / len(sub)),
        }
    return out


def escalation_curve(rows) -> list[dict]:
    """Escalate a ticket if the classifier's sentiment OR department confidence is below t.
    Reports what the classifier keeps; policy.py adds how the LLM does on the rest."""
    out = []
    for i in range(0, 100, 5):
        t = i / 100
        kept = [r for r in rows if min(r["sentiment_conf"], r["department_conf"]) >= t]
        both = [r["sentiment"] == r["t_sentiment"] and r["department"] == r["t_department"] for r in kept]
        out.append({
            "threshold": t,
            "escalation_rate": r4(1 - len(kept) / len(rows)),
            "kept": len(kept),
            "kept_both_correct": r4(sum(both) / len(both)) if both else None,
            "kept_sentiment": r4(sum(r["sentiment"] == r["t_sentiment"] for r in kept) / len(kept)) if kept else None,
            "kept_department":
                r4(sum(r["department"] == r["t_department"] for r in kept) / len(kept)) if kept else None,
        })
    return out


# --------------------------------------------------------------------------- #
def pick_run(conn, run_id, set_name):
    if run_id:
        return conn.execute("SELECT * FROM classifier_runs WHERE id = ?", (run_id,)).fetchone()
    return conn.execute(
        "SELECT * FROM classifier_runs WHERE question_set = ? ORDER BY id DESC LIMIT 1", (set_name,)
    ).fetchone()


def build_report(conn, run) -> dict:
    rows = [dict(r) for r in conn.execute(ROWS_SQL, (run["id"],)).fetchall()]
    lat = sorted(r["latency_ms"] for r in rows)
    other_runs = conn.execute(
        """SELECT r.id, r.question_set FROM classifier_runs r
           JOIN (SELECT question_set, MAX(id) id FROM classifier_runs GROUP BY question_set) m ON m.id = r.id"""
    ).fetchall()
    compare = {}
    for o in other_runs:
        sub = conn.execute(ROWS_SQL, (o["id"],)).fetchall()
        compare[o["question_set"]] = {
            "run_id": o["id"],
            "department_accuracy": r4(sum(r["department"] == r["t_department"] for r in sub) / len(sub)),
            "sentiment_accuracy": r4(sum(r["sentiment"] == r["t_sentiment"] for r in sub) / len(sub)),
        }
    return {
        "run": dict(run),
        "n_tickets": len(rows),
        "latency_ms": {"p50": r4(lat[len(lat) // 2]), "p95": r4(lat[int(len(lat) * 0.95)]),
                       "mean": r4(sum(lat) / len(lat))},
        "sentiment": choice_metrics(rows, "sentiment", "sentiment_conf", "t_sentiment", SENTIMENTS),
        "department": choice_metrics(rows, "department", "department_conf", "t_department", DEPARTMENTS),
        "urgency": urgency_metrics(rows),
        "churn_risk": binary_metrics(rows, "churn_p", "t_churn"),
        "refund_request": binary_metrics(rows, "refund_p", "t_refund"),
        "language_detection": {
            "accuracy": r4(sum(r["detected_lang"] == r["lang"] for r in rows) / len(rows)),
            "misses": [{"ticket_id": r["ticket_id"], "true": r["lang"], "detected": r["detected_lang"],
                        "script": r["detected_script"]} for r in rows if r["detected_lang"] != r["lang"]],
        },
        "by_language": breakdown(rows, "lang"),
        "by_difficulty": breakdown(rows, "difficulty"),
        "by_checkpoint": breakdown(rows, "checkpoint"),
        "question_set_comparison": compare,
        "escalation_curve": escalation_curve(rows),
        # Per-ticket data so the dashboard can recompute curves interactively.
        "tickets": [{k: r[k] for k in (
            "ticket_id", "lang", "difficulty", "channel", "checkpoint",
            "t_sentiment", "sentiment", "sentiment_conf",
            "t_department", "department", "department_conf",
            "t_urgency", "urgency_score", "t_churn", "churn_p", "t_refund", "refund_p")} for r in rows],
    }


def print_summary(rep: dict) -> None:
    pct = lambda x: "  n/a" if x is None else f"{x:6.1%}"  # noqa: E731
    run = rep["run"]
    print(f"\nEvaluation of run {run['id']} (question set {run['question_set']}, {run['device']}, "
          f"{rep['n_tickets']} tickets, p50 {rep['latency_ms']['p50']:.1f} ms)")

    print("\nChoice questions        accuracy  macro-F1   ECE")
    for q in ("sentiment", "department"):
        m = rep[q]
        print(f"  {q:<20} {pct(m['accuracy'])}   {pct(m['macro_f1'])}  {m['calibration']['ece']:.3f}")

    print("\nAccuracy of the tickets the classifier keeps, by confidence threshold (sentiment AND department):")
    print("  threshold  escalated  both-correct  sentiment  department")
    for p in rep["escalation_curve"]:
        if p["threshold"] in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            print(f"  {p['threshold']:>8.2f}   {pct(p['escalation_rate'])}      {pct(p['kept_both_correct'])}"
                  f"     {pct(p['kept_sentiment'])}     {pct(p['kept_department'])}")

    print("\nYes/no questions   base rate   AUC    P/R at 0.5        P/R at CV-best threshold")
    for q in ("churn_risk", "refund_request"):
        m = rep[q]
        a, b = m["at_0_5"], m["cv_best_f1"]
        print(f"  {q:<16} {pct(m['base_rate'])}    {m['auc']:.3f}  {pct(a['precision'])}/{pct(a['recall'])}"
              f"     {pct(b['precision'])}/{pct(b['recall'])}  (thresholds {sorted(set(b['thresholds_chosen']))})")

    u = rep["urgency"]
    print(f"\nUrgency: classifier's class {pct(u['class_accuracy'])} | majority baseline {pct(u['majority_baseline'])}"
          f" | tuned cut-offs {pct(u['tuned_cutoffs']['cv_accuracy'])} held-out "
          f"(t1={u['tuned_cutoffs']['t1']}, t2={u['tuned_cutoffs']['t2']}) | blocking AUC {u['blocking_auc']:.3f}")

    print("\nBy language:   " + "  ".join(f"{k}:{v['sentiment']:.0%}/{v['department']:.0%}"
                                          for k, v in rep["by_language"].items()) + "   (sentiment/department)")
    print("By difficulty: " + "  ".join(f"{k}:{v['sentiment']:.0%}(n={v['n']})"
                                          for k, v in rep["by_difficulty"].items()) + "   (sentiment)")
    print("Question sets: " + "  ".join(f"{k}: dept {v['department_accuracy']:.1%}"
                                          for k, v in rep["question_set_comparison"].items()))
    ld = rep["language_detection"]
    print(f"Language detection: {ld['accuracy']:.1%}; misses: "
          + ", ".join(f"#{m['ticket_id']} {m['true']}->{m['detected']}" for m in ld["misses"]))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate classifier predictions against ground truth")
    p.add_argument("--set", default="v1", help="question set (latest run of it is used)")
    p.add_argument("--run-id", type=int, default=None)
    args = p.parse_args()

    conn = connect()
    run = pick_run(conn, args.run_id, args.set)
    if run is None:
        sys.exit("No matching classifier run. Run: python -m pipeline.classifier")
    rep = build_report(conn, run)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print_summary(rep)
    print(f"\nFull report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
