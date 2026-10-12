# Triage Cascade: design, results and lessons

## Summary

I built a support-ticket triage pipeline in which a small classifier answers every ticket first and a generative LLM is called only when needed. The classifier tier is [Laya](https://huggingface.co/convaiinnovations/laya), a non-generative decision model: an encoder with a decision head that scores every answer option in one forward pass, at about 20 ms per ticket on a GPU. The LLM tier is Gemma 4 12B in LM Studio, at about 1.3 s per ticket. Everything runs on one desktop GPU and costs nothing.

All timings come from one machine (RTX 4080 Super, 16 GB; see the README). Your mileage will vary with hardware and server settings.

On 450 synthetic tickets in 9 languages:

- **Laya alone** gets sentiment right 72% of the time, department 70%, and catches 3% of churn threats.
- **Gemma alone** gets 83%, 97% and 100%, but takes about 70 times as long.
- **A routed policy** that sends 70% of tickets to Gemma reaches 81% / 91% / 67%. A cheaper one that sends 26% fixes most of the churn and refund problems.

The more useful results are about **where** the cheap model can be trusted, and about how easily the evaluation itself can mislead.

## 1. The question

Most LLM classification demos send every item to a large model. I wanted to measure the alternative:

> How much of the work can a cheap, fast model handle on its own, how do we know which tickets it can't handle, and what does sending those to an LLM actually buy?

Each ticket gets five answers: sentiment (positive, neutral or negative), department (billing, technical, account, sales or other), urgency (0–2), churn risk (yes/no) and refund request (yes/no).

## 2. System design

```
generate.py ──▶ tickets (SQLite) ──▶ classifier.py (Laya) ──▶ evaluate.py
                                          │
                                          ▼
                    graph.py (LangGraph): classifier → triage ─┬─▶ finalize
                                                               └─▶ llm (Gemma) ─▶ finalize
                                          │
                                          ▼
                              api/ (FastAPI) ──▶ web/ (React + TypeScript)
```

| Choice | Why |
| --- | --- |
| Laya as the classifier tier | Answers typed questions (choice, score, yes/no) in one forward pass and returns probabilities. It doesn't generate text, so there is nothing to parse. |
| Gemma 4 12B in LM Studio, thinking off | Fits a 16 GB GPU alongside Laya (Q4/Q5 GGUF, about 7–8 GB). Thinking took 3/4 of the time in the environment check and once used up the whole token budget, leaving an empty answer. |
| Labels first, then generation | The sampler picks the labels, and Gemma writes a ticket to match them. That gives ground truth without any manual labelling. |
| SQLite | One file, part of Python's standard library. Every run is versioned, so experiments can be compared later. |
| Cached answers per model | Laya and Gemma each label every ticket once. Policy sweeps then cost nothing, and the CLI and dashboard share one implementation (`policy.py`). |
| LangGraph | The flow is small, so an `if` statement would work today. The graph makes the routing explicit and inspectable. Adding a node later is a local change: a reply drafter, or a human-review interrupt. |

## 3. Data

- **Size and spread:** 450 tickets, 50 per language: English, Traditional Chinese, Japanese, Korean, Spanish, Brazilian Portuguese, French, German and Hindi. They were generated in 11 minutes with no failures.
- **Writing styles:** plain (244), informal (56), long (50), mixed sentiment (41), code-switching (36), hidden churn hints (14) and sarcasm (9).
- **Audit:**
  - An independent language detector agreed with every requested language except 5 Hindi tickets that Gemma wrote in romanized Hindi. That's realistic, so I kept them.
  - About 5% of low-urgency tickets contain urgent-sounding words, so the urgency labels are somewhat noisy.
- **Fixed during the audit:**
  - A JSON wrapper leaked into one ticket.
  - Some label combinations made Gemma write the wrong topic. An urgent "praise" ticket turned into a bug report, so the sampler no longer allows it.
  - The language mix was uneven, so languages are now assigned round-robin.

## 4. Findings

### 4.1 Laya's confidence ranks answers well but overstates them

| Laya confidence | Share of tickets | Sentiment correct |
| --- | --- | --- |
| any | 100% | 72% |
| ≥ 0.8 | 76% | 79% |
| ≥ 0.9 | 60% | 84% |
| ≥ 0.95 | 49% | 86% |

Answers given 0.7–0.8 confidence are right only about half the time. The expected calibration error (ECE) is 0.15 for sentiment and 0.20 for department. So the scores aren't honest probabilities, despite the model card's claim of calibration, but they **rank** answers well. A ranking is all an escalation threshold needs.

### 4.2 Accuracy hides a broken detector

Churn "accuracy" was 87%, because 86% of tickets have no churn threat. Recall at the default 0.5 cut-off was **3%** (2 of 61). The score still separates the classes (AUC 0.82), but no cut-off makes it good enough on its own: at 0.005, it reaches 70% recall at 30% precision. I used it as a **filter** for which tickets Gemma should check, not as the detector.

Refund is the opposite: AUC 0.98 and 93% recall at 0.5, but with false alarms on other billing complaints. A 0.7 cut-off gives 73% precision and 85% recall.

### 4.3 Some limits come from the labels, not the models

- **Urgency:** Laya's own class scored 25%, worse than always guessing "not urgent" (41%). Cut-offs on its score, tuned with 5-fold cross-validation, reach 48%. Gemma gets 54%. When very different models stall at the same number, the definition is the likely problem.
- **Neutral sentiment:** Gemma's sentiment errors are mostly on "neutral" tickets: 61 of the 174 neutral tickets were labelled something else. On mixed-sentiment tickets, even Gemma scores 46%.

### 4.4 Where the cheap model fails, and the LLM doesn't

| Style | Laya sentiment | Gemma sentiment |
| --- | --- | --- |
| plain | 79% | 87% |
| sarcasm | 22% (2 of 9) | 100% (9 of 9) |
| mixed | 34% | 46% |

For department, Laya's weakest languages are Korean (50%) and Japanese (62%). English is its strongest, at 82%. Gemma reaches 90% or more in every language.

### 4.5 Routing results

| Policy | Sent to Gemma | Sentiment | Department | Churn recall | Refund precision | Time, 450 tickets |
| --- | --- | --- | --- | --- | --- | --- |
| Laya only | 0% | 72% | 70% | 3% | 71% | 9 s |
| Budget: churn + refund checks only | 26% | 73% | 76% | 52% | 95% | 2.8 min |
| Balanced: confidence < 0.9 + both checks | 70% | 81% | 91% | 67% | 89% | 7.1 min |
| Gemma only | 100% | 83% | 97% | 100% | 80% | 9.9 min |

- **The budget policy** fixes most of the churn and refund problems with a quarter of the Gemma calls.
- **The mix beats Gemma alone on refund precision.** Laya's strict cut-off settles the easy cases, and Gemma only sees the borderline ones.
- **The time saved depends on how expensive Gemma is.** With Gemma this fast, routing saves about 30%. With a 10 s "thinking" model, the same policy saves 69% of a 75-minute batch. The dashboard has a setting for this.

### 4.6 Smaller results worth knowing

- **Detect language with the router, not a question.** Asking Laya "which language is this?" scored 2/10 in the environment check. The router's own detection was right for 445 of 450 tickets.
- **Romanized Hindi breaks detection.** Four of the five romanized Hindi tickets were detected as Italian, and one as no language.
- **Shorter option wording works better for Laya.** Longer, more explanatory department descriptions lowered department accuracy from 70% to 61%.

## 5. Evaluation pitfalls I hit

1. **Leakage through shared wording.** Gemma's first labelling prompt reused the generator's department descriptions, and it scored 99%. With the same short definitions Laya gets, it scored 97%. The leak was small, but measuring it showed that the comparison is fair now.
2. **The LLM labels text it wrote itself.** Gemma wrote the tickets, so its scores are likely optimistic. No prompt change fixes that; it needs human-labelled or real data (section 7).
3. **Tuning on the test set.** Thresholds for urgency, churn and refund are chosen by 5-fold cross-validation and scored on held-out tickets.
4. **Synthetic data is cleaner than real tickets.** Read these numbers as a controlled benchmark, not production performance.

## 6. Engineering notes

- **Windows, no WSL.** LM Studio, CUDA PyTorch (Python 3.14, cu130) and Node all work natively. The pitfall was `pip` pulling the CPU-only PyTorch build.
- **VRAM budget on 16 GB.** Laya peaked at about 5 GB with all three checkpoints loaded (the pipeline loads only two), Gemma 12B at Q4/Q5 needs about 7–8.5 GB, and the desktop about 1 GB.
- **Thinking mode** caused an empty answer at a 250-token limit, because the hidden reasoning used up the whole budget. It's off for batch work and kept as a possible third tier for hard tickets.
- **Live triage** keeps Laya and the Gemma client loaded between requests. Loading Laya per request would cost about 9 s each time.

## 7. What I'd do next, or in production

1. **Build a human-labelled gold set.** 150–200 real or hand-checked tickets, labelled by two people, would give honest numbers, show how far annotators disagree with each other (the real ceiling for urgency and neutral), and remove the self-grading bias.
2. **Calibrate Laya.** Fit temperature scaling or isotonic regression on a held-out split, so a threshold means the same thing across questions and languages.
3. **Set thresholds per language and per question.** Korean and Japanese need a lower bar for escalation than English.
4. **Escalate more carefully.** Make thinking a third tier only for sarcasm, mixed sentiment and churn candidates, and measure whether it beats non-thinking Gemma there.
5. **Learn from the escalations.** Gemma's answers on escalated tickets are free training data to fine-tune or adapt the small model (distillation), shrinking the escalation rate over time.
6. **Operate it.** Run the LLM tier behind a queue, monitor the escalation rate and confidence over time (a rising rate means drift), run in shadow mode before switching, redact personal data before the LLM, and set a cost budget per day.
7. **Guard the reply drafts.** In the environment check, Gemma's German reply promised "our technical team is already working on it", which nothing supported. Drafts need a rule against unsupported promises and a human approval step.
