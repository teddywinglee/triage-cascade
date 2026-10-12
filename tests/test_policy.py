import pytest

from pipeline.policy import CLASSIFIER_ONLY, LLM_ONLY, Policy, escalation_reasons, evaluate_policy, final_answer


def pred(**kw):
    """A classifier prediction with confident, harmless defaults."""
    base = {"ticket_id": 1, "sentiment": "neutral", "sentiment_conf": 0.99, "department": "sales",
            "department_conf": 0.99, "urgency_score": 1.0, "churn_p": 0.0, "refund_p": 0.0, "latency_ms": 20.0}
    return {**base, **kw}


LLM = {"sentiment": "negative", "department": "billing", "urgency": 2, "churn_risk": 1,
       "refund_request": 1, "summary": "s", "seconds": 1.5, "ok": 1}


def test_confident_ticket_is_not_escalated():
    assert escalation_reasons(pred(), Policy()) == []


@pytest.mark.parametrize("kw, reason", [
    ({"department_conf": 0.5}, "low_confidence"),
    ({"churn_p": 0.01}, "churn_screen"),
    ({"refund_p": 0.6}, "refund_check"),
])
def test_each_rule_escalates(kw, reason):
    assert reason in escalation_reasons(pred(**kw), Policy())


def test_churn_screen_skips_positive_and_optionally_neutral_tickets():
    assert escalation_reasons(pred(sentiment="positive", churn_p=0.5), Policy()) == []
    assert escalation_reasons(pred(churn_p=0.5), Policy(churn_screen_neutral=False)) == []


def test_baselines():
    assert escalation_reasons(pred(department_conf=0.1, churn_p=0.9), CLASSIFIER_ONLY) == []
    assert escalation_reasons(pred(), LLM_ONLY) == ["always"]


def test_final_answer_uses_llm_only_when_escalated_and_successful():
    p = Policy()
    assert final_answer(pred(), LLM, [], p)["source"] == "classifier"
    assert final_answer(pred(), LLM, ["low_confidence"], p)["source"] == "llm"
    assert final_answer(pred(), {**LLM, "ok": 0}, ["low_confidence"], p)["source"] == "classifier"


def test_classifier_urgency_uses_score_cutoffs():
    p = Policy(urgency_t1=1.0, urgency_t2=2.0)
    assert [final_answer(pred(urgency_score=s), None, [], p)["urgency"] for s in (0.5, 1.5, 2.5)] == [0, 1, 2]


def test_evaluate_policy_counts_escalations_and_cost():
    truth = {"t_sentiment": "negative", "t_department": "billing", "t_urgency": 2, "t_churn": 1, "t_refund": 1}
    rows = [pred(ticket_id=1, **truth), pred(ticket_id=2, department_conf=0.5, **truth)]
    m = evaluate_policy(rows, {1: LLM, 2: LLM}, Policy(churn_screen=None, refund_check=None))
    assert m["escalated"] == 1 and m["reasons"] == {"low_confidence": 1}
    assert m["department"] == 0.5  # only the escalated ticket gets the LLM's (correct) answer
    assert m["seconds_per_ticket"] == pytest.approx((0.02 * 2 + 1.5) / 2, abs=1e-4)
