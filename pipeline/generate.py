"""
Generate labelled, multilingual support tickets with a local LLM.

The labels are chosen FIRST (by a seeded sampler with business-logic rules), then
the LLM is asked to write a ticket that matches them. The labels are therefore
ground truth for evaluating both tiers later.

Usage (project root, venv active):
    python -m pipeline.generate --n 20            # small test batch
    python -m pipeline.generate --n 450           # full dataset (appends, skips duplicates)
    python -m pipeline.generate --n 10 --dry-run  # show sampled label specs, no LLM calls
    python -m pipeline.generate --stats           # summary of what's in the database
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass

from pydantic import BaseModel, Field

from . import config
from .db import connect

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --------------------------------------------------------------------------- #
# Label space
# --------------------------------------------------------------------------- #
LANGUAGES = {
    "en": "English",
    "zh": "Traditional Chinese (as written in Taiwan)",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "pt": "Brazilian Portuguese",
    "fr": "French",
    "de": "German",
    "hi": "Hindi (Devanagari script)",
}
DEPARTMENTS = {
    "billing": "a charge, invoice, payment method, or refund",
    "technical": "a bug, crash, sync problem, outage, error message, or slowness",
    "account": "login, password, email address, profile, or account settings",
    "sales": "upgrading, choosing a plan, pricing, discounts, or buying more seats",
    "other": "general feedback, praise, or a feature suggestion",
}
CHANNELS = {
    "email": "a support email (may include a greeting and sign-off)",
    "chat": "a short live-chat message",
    "app_review": "an app store review",
}
URGENCY = {
    0: "not urgent at all (no time pressure, no words like urgent/ASAP)",
    1: "would like it handled soon, but NOT urgent or blocking (no words like urgent/ASAP/immediately)",
    2: "blocking their work right now, clearly urgent",
}
DIFFICULTY = {
    "plain": "Write it plainly and naturally.",
    "sarcasm": "The customer is sarcastic: the words sound positive but the meaning is clearly negative.",
    "mixed": "The customer mentions both something they like and something they dislike, balanced overall.",
    "implicit_churn": ("The customer hints they may leave (e.g. comparing to another app, reconsidering renewal) "
                       "without saying 'cancel' outright."),
    "code_switch": "Mix in a few English words or product terms, as bilingual users often do.",
    "informal": "Very informal: slang, abbreviations, a typo or two, little punctuation.",
    "long": "Longer message (4-6 sentences) with some background detail before the main point.",
}


@dataclass
class Spec:
    lang: str
    channel: str
    department: str
    sentiment: str
    urgency: int
    churn_risk: bool
    refund_request: bool
    difficulty: str


def sample_spec(rng: random.Random, lang: str | None = None) -> Spec:
    """Sample a coherent label combination (not uniform: real tickets have structure)."""
    lang = lang or rng.choice(list(LANGUAGES))
    channel = rng.choices(list(CHANNELS), weights=[0.45, 0.35, 0.20])[0]
    department = rng.choices(list(DEPARTMENTS), weights=[0.25, 0.30, 0.15, 0.15, 0.15])[0]

    sent_weights = {  # positive, neutral, negative
        "billing": [0.10, 0.35, 0.55],
        "technical": [0.05, 0.25, 0.70],
        "account": [0.05, 0.60, 0.35],
        "sales": [0.25, 0.65, 0.10],
        "other": [0.55, 0.25, 0.20],
    }[department]
    sentiment = rng.choices(["positive", "neutral", "negative"], weights=sent_weights)[0]

    urgency = rng.choices([0, 1, 2], weights={
        "positive": [0.90, 0.10, 0.00],
        "neutral": [0.50, 0.40, 0.10],
        "negative": [0.15, 0.40, 0.45],
    }[sentiment])[0]
    # Feedback/praise and pre-sales questions are never "blocking" in practice;
    # forcing it makes the LLM invent a bug or payment error (label drift).
    if department in ("other", "sales"):
        urgency = min(urgency, 1)
    if department == "other" and sentiment == "negative":
        urgency = 0

    refund_request = department == "billing" and sentiment != "positive" and rng.random() < 0.5
    churn_risk = sentiment == "negative" and rng.random() < 0.35

    options = ["plain"] * 5 + ["informal", "long"]
    if lang != "en":
        options.append("code_switch")
    if sentiment == "negative":
        options.append("sarcasm")
    if sentiment == "neutral":
        options += ["mixed", "mixed"]
    if churn_risk:
        options += ["implicit_churn", "implicit_churn"]
    difficulty = rng.choice(options)

    return Spec(lang, channel, department, sentiment, urgency, churn_risk, refund_request, difficulty)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
class GeneratedTicket(BaseModel):
    text: str = Field(description="The customer's message only, in the requested language.")


SYSTEM_PROMPT = f"""You write realistic synthetic customer messages for testing a support-ticket classifier.
Company context: {config.PRODUCT_BLURB}
Rules:
- Output ONLY the customer's message, never a reply from the company.
- Match every requested attribute exactly; the attributes are ground-truth labels.
- Do not mention the attribute names (e.g. never write "urgency" or "sentiment").
- The message must clearly belong to the requested topic only; do not drift into another topic
  (e.g. a sales question must not describe a bug or a failed payment).
- Vary names, details, wording and length; avoid starting every message the same way.
- Respond with JSON: {{"text": "<the message>"}}"""


def build_prompt(spec: Spec) -> str:
    lines = [
        f"Language: {LANGUAGES[spec.lang]}. Write the whole message in this language.",
        f"Channel: {CHANNELS[spec.channel]}.",
        f"Topic (department): {spec.department}, i.e. {DEPARTMENTS[spec.department]}.",
        f"Overall sentiment: {spec.sentiment}.",
        f"Urgency: {URGENCY[spec.urgency]}.",
        "The customer DOES threaten or strongly signal leaving / cancelling / switching to a competitor."
        if spec.churn_risk else "The customer does NOT threaten to leave or cancel.",
        "The customer DOES ask for a refund or money back."
        if spec.refund_request else "The customer does NOT ask for a refund or money back.",
        f"Style: {DIFFICULTY[spec.difficulty]}",
    ]
    return "Write one customer message with these attributes:\n- " + "\n- ".join(lines)


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #
def make_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=config.LLM_BASE_URL, api_key="lm-studio", model=config.LLM_MODEL,
        temperature=0.9, max_tokens=1500, timeout=300,
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_text(raw: str) -> str | None:
    """Accept JSON {"text": ...}; fall back to the raw text if the model ignored JSON."""
    raw = raw.strip()
    m = _JSON_RE.search(raw)
    if m:
        try:  # strict=False allows literal newlines inside JSON strings
            return GeneratedTicket.model_validate(json.loads(m.group(0), strict=False)).text.strip()
        except Exception:  # noqa: BLE001
            return None  # looked like JSON but broken: retry rather than store garbage
    raw = re.sub(r"^```(?:json)?|```$", "", raw).strip()
    return raw or None


def generate_one(llm, spec: Spec, retries: int = 2) -> tuple[str | None, float]:
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_prompt(spec))]
    start = time.perf_counter()
    for _ in range(retries + 1):
        try:
            msg = llm.invoke(messages)
        except Exception as e:  # noqa: BLE001
            print(f"    ! LLM error: {type(e).__name__}: {e}")
            time.sleep(2)
            continue
        text = parse_text(msg.content or "")
        if text and len(text) >= 10:
            return text, time.perf_counter() - start
    return None, time.perf_counter() - start


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def print_stats(conn) -> None:
    rows = conn.execute("SELECT * FROM tickets").fetchall()
    print(f"\nTickets in {config.DB_PATH.name}: {len(rows)}")
    if not rows:
        return
    for col in ["lang", "sentiment", "department", "urgency", "churn_risk", "refund_request", "difficulty", "channel"]:
        c = Counter(r[col] for r in rows)
        print(f"  {col:<15} " + ", ".join(f"{k}={v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate labelled synthetic tickets into SQLite")
    p.add_argument("--n", type=int, default=20, help="number of tickets to generate")
    p.add_argument("--seed", type=int, default=None, help="random seed (default: time-based)")
    p.add_argument("--dry-run", action="store_true", help="print sampled specs + prompt, no LLM")
    p.add_argument("--stats", action="store_true", help="print dataset summary and exit")
    args = p.parse_args()

    conn = connect()
    if args.stats:
        print_stats(conn)
        return

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    # Round-robin languages so each gets an equal share, then shuffle the order.
    langs = [list(LANGUAGES)[i % len(LANGUAGES)] for i in range(args.n)]
    rng.shuffle(langs)
    specs = [sample_spec(rng, lang) for lang in langs]

    if args.dry_run:
        for s in specs:
            print(json.dumps(asdict(s), ensure_ascii=False))
        print("\nExample prompt:\n" + build_prompt(specs[0]))
        return

    llm = make_llm()
    run_id = conn.execute(
        "INSERT INTO generation_runs (model, seed, requested) VALUES (?, ?, ?)",
        (config.LLM_MODEL, seed, args.n),
    ).lastrowid
    conn.commit()
    print(f"Run {run_id}: generating {args.n} tickets with {config.LLM_MODEL} @ {config.LLM_BASE_URL} (seed {seed})")

    ok = dup = fail = 0
    t0 = time.perf_counter()
    for i, spec in enumerate(specs, 1):
        text, secs = generate_one(llm, spec)
        if text is None:
            fail += 1
            status = "FAILED"
        else:
            cur = conn.execute(
                """INSERT OR IGNORE INTO tickets
                   (run_id, text, channel, lang, sentiment, department, urgency,
                    churn_risk, refund_request, difficulty, gen_seconds)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, text, spec.channel, spec.lang, spec.sentiment, spec.department,
                 spec.urgency, int(spec.churn_risk), int(spec.refund_request), spec.difficulty, round(secs, 2)),
            )
            conn.commit()
            if cur.rowcount:
                ok += 1
                status = "ok"
            else:
                dup += 1
                status = "duplicate"
        elapsed = time.perf_counter() - t0
        eta = elapsed / i * (len(specs) - i)
        preview = (text or "").replace("\n", " ")[:70]
        print(f"  [{i:>4}/{len(specs)}] {secs:5.1f}s {spec.lang} {spec.sentiment:<8} {spec.department:<9} "
              f"{spec.difficulty:<14} {status:<9} ETA {eta/60:4.1f}m  {preview}")

    print(f"\nDone in {(time.perf_counter()-t0)/60:.1f} min: {ok} saved, {dup} duplicates, {fail} failed.")
    print_stats(conn)


if __name__ == "__main__":
    main()
