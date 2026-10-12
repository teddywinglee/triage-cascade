"""
Environment check: run this before the pipeline.

Checks, in order:
  1. Environment     - Python, PyTorch, CUDA / GPU.
  2. Classifier tier - Laya Router: load time, per-ticket latency, answers vs. expected
                       labels on 10 hand-written multilingual tickets, GPU memory.
  3. LLM tier        - an OpenAI-compatible server (e.g. LM Studio) via LangChain: one
                       summary call, latency and rough tokens/sec.

Results are written to data/env_check.json, including Laya's raw output structure.

Usage (from the project root, venv active):
    python scripts/check_env.py
    python scripts/check_env.py --device cpu          # force CPU for Laya
    python scripts/check_env.py --skip-llm            # classifier only
    python scripts/check_env.py --llm-url http://localhost:1234/v1 --llm-model gemma-4-12b-it

Environment variable overrides: LLM_BASE_URL, LLM_MODEL.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from statistics import mean, median

# Windows consoles (e.g. cp950 in Taiwan) cannot print Hindi/Korean/etc. by default.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "env_check.json"

DEFAULT_LLM_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-12b-it")

# --------------------------------------------------------------------------- #
# Questions asked of every ticket (one forward pass for all of them).
# --------------------------------------------------------------------------- #
QUESTIONS = {
    "sentiment": {
        "type": "choice",
        "instructions": "What is the overall sentiment of the customer's message?",
        "criteria": {
            "positive": "happy, satisfied, praising, thankful",
            "neutral": "factual question or request, mixed or no clear emotion",
            "negative": "unhappy, frustrated, angry, complaining (including sarcasm)",
        },
    },
    "language": {
        "type": "choice",
        "instructions": "Which language is the message written in?",
        "criteria": {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "pt": "Portuguese",
            "fr": "French",
            "de": "German",
            "hi": "Hindi",
            "other": "any other language",
        },
    },
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this message?",
        "criteria": {
            "billing": "invoices, payments, charges, refunds",
            "technical": "bugs, crashes, outages, errors, performance",
            "account": "login details, profile, email, settings changes",
            "sales": "upgrades, plans, pricing, discounts for new purchases",
            "other": "feedback, praise, anything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["not urgent", "soon", "blocking"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the customer threaten to cancel, leave, or switch to a competitor?",
    },
    "refund_request": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund or money back?",
    },
}

# --------------------------------------------------------------------------- #
# Hand-written tickets with the labels we expect. Includes edge cases:
# sarcasm (#10), mixed sentiment (#9), a thank-you that mentions a refund (#6).
# --------------------------------------------------------------------------- #
TICKETS = [
    {"id": 1, "text": "I've been charged twice for my subscription this month. Refund the extra charge today or I'm cancelling.",
     "expected": {"sentiment": "negative", "language": "en", "department": "billing", "churn_risk": True, "refund_request": True}},
    {"id": 2, "text": "App 更新後一直閃退，完全沒辦法登入，明天要開會急用，請盡快處理！",
     "expected": {"sentiment": "negative", "language": "zh", "department": "technical", "churn_risk": False, "refund_request": False}},
    {"id": 3, "text": "新しいダッシュボード機能、とても使いやすいです。チームのみんなも気に入っています。",
     "expected": {"sentiment": "positive", "language": "ja", "department": "other", "churn_risk": False, "refund_request": False}},
    {"id": 4, "text": "¿Podrían decirme cómo cambiar el correo electrónico de mi cuenta? Gracias.",
     "expected": {"sentiment": "neutral", "language": "es", "department": "account", "churn_risk": False, "refund_request": False}},
    {"id": 5, "text": "Seit drei Tagen funktioniert der Export nicht. Wenn das so weitergeht, wechseln wir zur Konkurrenz.",
     "expected": {"sentiment": "negative", "language": "de", "department": "technical", "churn_risk": True, "refund_request": False}},
    {"id": 6, "text": "Merci pour le remboursement rapide, le service client a été parfait !",
     "expected": {"sentiment": "positive", "language": "fr", "department": "billing", "churn_risk": False, "refund_request": False}},
    {"id": 7, "text": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
     "expected": {"sentiment": "negative", "language": "hi", "department": "billing", "churn_risk": False, "refund_request": True}},
    {"id": 8, "text": "요금제를 업그레이드하고 싶은데 연간 결제 할인도 있나요?",
     "expected": {"sentiment": "neutral", "language": "ko", "department": "sales", "churn_risk": False, "refund_request": False}},
    {"id": 9, "text": "O aplicativo é bom, mas a sincronização às vezes demora um pouco.",
     "expected": {"sentiment": "neutral", "language": "pt", "department": "technical", "churn_risk": False, "refund_request": False}},
    {"id": 10, "text": "Oh great, another update that deletes all my saved settings. Love it.",
     "expected": {"sentiment": "negative", "language": "en", "department": "technical", "churn_risk": False, "refund_request": False}},
]

GRADED = ["sentiment", "language", "department", "churn_risk", "refund_request"]


# --------------------------------------------------------------------------- #
def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check_environment() -> dict:
    section("1. Environment")
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            info["gpu_mem_free_gb"] = round(free / 1024**3, 2)
            info["gpu_mem_total_gb"] = round(total / 1024**3, 2)
    except ImportError:
        info["torch"] = None
    for k, v in info.items():
        print(f"  {k:<18} {v}")
    if info.get("torch") and not info.get("cuda_available"):
        print("\n  ! PyTorch has no CUDA. Laya will run on CPU. See README 'Run it'")
        print("    to install the CUDA build of PyTorch.")
    return info


def extract(answer: dict, qtype: str):
    """Pull the headline value out of a Laya answer dict."""
    return answer.get(qtype)


def find_confidence(answer: dict, qtype: str):
    """Laya returns `answer_confidence` (probability of the chosen answer) on every
    answer type. For noul we report max(p, 1-p) so 'confident no' also counts."""
    if qtype == "noul":
        v = answer.get("noul")
        return None if v is None else round(max(v, 1 - v), 4)
    return answer.get("answer_confidence")


# Non-Latin scripts: Laya's router reports the script but not the language.
SCRIPT_TO_LANG = {"han": "zh", "kana": "ja", "hangul": "ko", "devanagari": "hi",
                  "cyrillic": "ru", "arabic": "ar", "thai": "th", "hebrew": "he"}


def router_language(result: dict):
    det = (result.get("routing") or {}).get("detection") or {}
    return det.get("language") or SCRIPT_TO_LANG.get(det.get("script"))


def run_laya(device: str) -> dict:
    section(f"2. Classifier tier: Laya Router (device={device})")
    import torch
    from laya import Router

    t0 = time.perf_counter()
    router = Router(preload=True, device=device)
    load_s = time.perf_counter() - t0
    print(f"  Router loaded (all checkpoints, incl. first-time download) in {load_s:.1f}s")

    # Warm-up: first call pays for kernel compilation / caching.
    router.predict(TICKETS[0]["text"], QUESTIONS)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rows, latencies, correct = [], [], {q: 0 for q in GRADED}
    for t in TICKETS:
        s = time.perf_counter()
        result = router.predict(t["text"], QUESTIONS)
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - s) * 1000
        latencies.append(ms)

        answers = result.get("answers", {})
        pred, conf = {}, {}
        for q, spec in QUESTIONS.items():
            a = answers.get(q, {})
            pred[q] = extract(a, spec["type"])
            conf[q] = find_confidence(a, spec["type"])
        # Language: compare Laya's "language" question with the router's own detection.
        pred["language_question"] = pred["language"]
        pred["language"] = router_language(result)

        hits = {}
        for q in GRADED:
            exp = t["expected"][q]
            got = pred[q]
            ok = (got is not None and (got > 0.5) == exp) if isinstance(exp, bool) else (got == exp)
            hits[q] = ok
            correct[q] += ok

        rows.append({
            "id": t["id"], "text": t["text"], "latency_ms": round(ms, 1),
            "routing": result.get("routing"), "predicted": pred, "confidence": conf,
            "expected": t["expected"], "correct": hits, "raw": result,
        })

        marks = " ".join(f"{q[:4]}{'✓' if hits[q] else '✗'}" for q in GRADED)
        model = (result.get("routing") or {}).get("model", "?")
        print(f"  #{t['id']:<2} {ms:7.1f} ms  [{model:<12}] {marks}  "
              f"sent={pred['sentiment']} dept={pred['department']} urg={pred['urgency']}")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None

    lq = sum(r["predicted"]["language_question"] == r["expected"]["language"] for r in rows)
    print("\n  Accuracy on 10 hand-labelled tickets:")
    print(f"    {'language (asked as a question)':<15} {lq}/{len(TICKETS)}   <- for comparison; graded 'language' uses router detection")
    for q in GRADED:
        print(f"    {q:<15} {correct[q]}/{len(TICKETS)}")
    print(f"\n  Latency per ticket ({len(QUESTIONS)} questions, one pass): "
          f"mean {mean(latencies):.1f} ms, median {median(latencies):.1f} ms, max {max(latencies):.1f} ms")
    if peak_gb is not None:
        print(f"  Peak GPU memory allocated by PyTorch: {peak_gb:.2f} GB")

    print("\n  Raw answer structure for ticket #1 (Laya's output keys):")
    print("  " + json.dumps(rows[0]["raw"], ensure_ascii=False, indent=2, default=str).replace("\n", "\n  "))

    return {
        "device": device, "load_seconds": round(load_s, 2),
        "latency_ms": {"mean": round(mean(latencies), 1), "median": round(median(latencies), 1), "max": round(max(latencies), 1)},
        "accuracy": {q: f"{correct[q]}/{len(TICKETS)}" for q in GRADED},
        "peak_gpu_gb": None if peak_gb is None else round(peak_gb, 2),
        "tickets": rows,
    }


def run_llm(base_url: str, model: str) -> dict:
    section(f"3. LLM tier: OpenAI-compatible server via LangChain ({model} @ {base_url})")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("  ! langchain-openai not installed: pip install langchain-openai")
        return {"ok": False, "error": "langchain-openai missing"}

    llm = ChatOpenAI(base_url=base_url, api_key="lm-studio", model=model,
                     temperature=0.2, max_tokens=1500, timeout=300)
    ticket = TICKETS[4]["text"]  # German churn-risk ticket
    prompt = (
        "You are a support team lead. For the customer ticket below, reply in English with:\n"
        "1. A one-sentence summary.\n"
        "2. A short, polite reply draft written in the customer's own language.\n\n"
        f"Ticket:\n{ticket}"
    )
    try:
        s = time.perf_counter()
        msg = llm.invoke(prompt)
        secs = time.perf_counter() - s
    except Exception as e:  # noqa: BLE001 - environment check: report anything
        print(f"  ! LLM call failed: {type(e).__name__}: {e}")
        print("    Check LM Studio: server started, model loaded, 'Serve on Local Network' on,")
        print("    and the URL ends with /v1.")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    usage = getattr(msg, "usage_metadata", None) or {}
    out_tok = usage.get("output_tokens")
    tps = (out_tok / secs) if out_tok else None
    print(f"  Response in {secs:.1f}s" + (f", {out_tok} output tokens ≈ {tps:.1f} tok/s" if tps else ""))
    reasoning = (msg.additional_kwargs or {}).get("reasoning_content") or ""
    if reasoning:
        print(f"  (model also produced {len(reasoning)} chars of hidden reasoning)")
    if not msg.content.strip():
        print("  ! Empty answer: the model likely spent all tokens thinking. Turn off")
        print("    'thinking'/reasoning for this model in LM Studio, or raise max_tokens.")
    print("\n  " + msg.content.strip().replace("\n", "\n  "))
    return {"ok": True, "seconds": round(secs, 2), "output_tokens": out_tok,
            "tokens_per_second": None if tps is None else round(tps, 1), "content": msg.content,
            "reasoning_chars": len(reasoning)}


def main() -> None:
    p = argparse.ArgumentParser(description="Check the environment: PyTorch/CUDA, Laya and the LLM server")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--skip-laya", action="store_true")
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    args = p.parse_args()

    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "environment": check_environment()}

    device = args.device
    if device == "auto":
        device = "cuda" if report["environment"].get("cuda_available") else "cpu"

    if not args.skip_laya:
        report["laya"] = run_laya(device)
    if not args.skip_llm:
        report["llm"] = run_llm(args.llm_url, args.llm_model)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    section("Done")
    print(f"  Full results written to {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
