"""Question sets asked of the classifier. Versioned so wording changes can be compared on the same tickets."""

SENTIMENT = {
    "type": "choice",
    "instructions": "What is the overall sentiment of the customer's message?",
    "criteria": {
        "positive": "happy, satisfied, praising, thankful",
        "neutral": "factual question or request, mixed or no clear emotion",
        "negative": "unhappy, frustrated, angry, complaining (including sarcasm)",
    },
}
URGENCY = {
    "type": "score",
    "instructions": "How urgent is this message?",
    "criteria": ["not urgent", "soon", "blocking"],
}
CHURN = {
    "type": "noul",
    "instructions": "Does the customer threaten to cancel, leave, or switch to a competitor?",
}
REFUND = {
    "type": "noul",
    "instructions": "Is the customer asking for a refund or money back?",
}

# v1: short option wording, as first tried in scripts/check_env.py (department scored 6/10 there).
DEPARTMENT_V1 = {
    "type": "choice",
    "instructions": "Which department should handle this message?",
    "criteria": {
        "billing": "invoices, payments, charges, refunds",
        "technical": "bugs, crashes, outages, errors, performance",
        "account": "login details, profile, email, settings changes",
        "sales": "upgrades, plans, pricing, discounts for new purchases",
        "other": "feedback, praise, anything else",
    },
}

# v2: longer, sharper boundaries for the confusions seen in that check
# (sales vs billing, praise mislabelled as technical). It scored worse: 61% vs. 70%.
DEPARTMENT_V2 = {
    "type": "choice",
    "instructions": ("Which team should handle this customer message? "
                     "Pick the team that owns the customer's main problem or request."),
    "criteria": {
        "billing": ("a problem with money already charged or owed: "
                    "wrong or double charge, invoice, receipt, failed payment, refund"),
        "technical": ("the app itself misbehaves: "
                      "bug, crash, sync failure, error message, slowness, export not working"),
        "account": "access to or details of the account: login, password reset, changing email, profile or settings",
        "sales": ("the customer wants to buy or change plan: "
                  "upgrade, pricing question, discount, adding seats; nothing is broken"),
        "other": "praise, general feedback or a feature idea, with nothing broken and nothing to buy",
    },
}

QUESTION_SETS = {
    "v1": {"sentiment": SENTIMENT, "department": DEPARTMENT_V1, "urgency": URGENCY,
           "churn_risk": CHURN, "refund_request": REFUND},
    "v2": {"sentiment": SENTIMENT, "department": DEPARTMENT_V2, "urgency": URGENCY,
           "churn_risk": CHURN, "refund_request": REFUND},
}
