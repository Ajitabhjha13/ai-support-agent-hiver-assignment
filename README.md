# AI Customer-Support Agent · Hiver SDE Intern Assignment

> **Note:** This is my submission for the **Hiver SDE Intern take-home assignment**, not a personal product. The problem statement came from Hiver. The design, code, labelled data and evaluation are my own work.

An AI agent that handles customer messages sent to **@AppleSupport** on Twitter, built on the Kaggle *Customer Support on Twitter* dataset. For each incoming message it:

1. **Classifies the intent** into one of 9 categories derived from the brand's real traffic.
2. **Drafts a reply** grounded in how AppleSupport actually answered similar messages in the past (TF-IDF retrieval over historical replies).
3. **Decides whether to auto-handle or escalate** to a human, and gives a reason.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)

---

## Results

All systems were evaluated on a **220-example golden set that I labelled by hand**. The LLM system was run on a 20-example subsample because of free-tier Gemini rate limits.

| System | Intent accuracy | Escalation accuracy | Escalation F1 |
|---|---|---|---|
| Trivial baseline (always predict the majority class), n=220 | 0.245 | 0.627 | n/a |
| TF-IDF + Logistic Regression (5-fold CV), n=220 | 0.473 | 0.709 | 0.515 |
| LLM agent, first prompt, n=20 | 0.550 | 0.450 | 0.560 |
| **LLM agent, revised prompt, n=20** | **0.650** | **0.900** | **0.875** |

- **Reply quality:** 4.92 / 5, scored by an LLM judge.
- **Judge reliability:** the judge was checked against 24 replies I scored by hand, with Pearson r = 0.967 and a mean absolute error of 0.25.

### What I learned from debugging the agent

The first escalation prompt scored **worse than the trivial baseline**: it escalated 85% of messages, while only 40% truly needed a human. The `escalation_reason` output showed why. Ordinary sarcastic venting about known bugs was being treated as an angry customer who needed a human. I rewrote the prompt so that frustration about a known issue is *not* an escalation trigger, and added four worked examples. Escalation accuracy went from **45% to 90%** on the same examples.

### Honest caveats

- The revised prompt was tuned and re-scored on the **same 20 examples**, so 90% is not a held-out number.
- With n=20, the 95% confidence interval is roughly ±13 points.
- Each message is handled on its own, without earlier turns of the thread. This is the biggest simplification in the project.

Full analysis (top 5 failure modes, what's misleading about the headline numbers, next steps): [`report.md`](report.md). Design decisions and trade-offs: [`decision_log.md`](decision_log.md).

---

## How it works

```
customer message
      │
      ▼
┌──────────────────┐   9 intents: battery_performance, device_bug, software_update_issue,
│ classify_intent  │   account_security, order_shipping, billing_refund, howto_info,
└────────┬─────────┘   praise_feedback, other_unclear
         ▼
┌──────────────────┐   TF-IDF retrieval of the most similar historical
│   draft_reply    │   (customer message → AppleSupport reply) pairs,
└────────┬─────────┘   used to ground the LLM's draft in the brand's real behaviour
         ▼
┌──────────────────┐   escalate if account or money data is involved, the customer is
│ decide_escalation│   abusive to the agent, the text is non-English, or context is missing
└──────────────────┘
```

## Project structure

```
src/
  data_prep.py      # raw twcs.csv → brand-filtered (customer_msg, brand_reply) pairs
  retrieval.py      # TF-IDF grounding retriever
  agent.py          # the 3-step agent: classify → draft → escalate
  llm_client.py     # provider wrapper: Google Gemini (free tier) or OpenAI
  baselines.py      # trivial and TF-IDF + Logistic Regression baselines (no LLM needed)
  eval/
    build_golden_set.py         # stratified sampling → draft labels
    finalize_golden_set.py      # human-reviewed corrections → final golden set
    build_judge_calibration.py  # hand-scored reply-quality set
    calibrate_judge.py          # LLM judge vs. human agreement
    llm_judge.py                # rubric-based reply-quality judge
    run_eval.py                 # end-to-end evaluation of the agent
data/
  pairs.csv                       # 20k (customer_msg, brand_reply) pairs, seeded subsample
  golden/golden_eval_set.csv      # 220 hand-labelled examples
  golden/golden_labeling_notes.md # sampling method and correction log
  golden/judge_calibration.csv    # 24 hand-scored replies
scripts/
  smoke_test.py     # runs the whole pipeline with a fake LLM (no API key needed)
report.md           # results, failure analysis, caveats
decision_log.md     # non-obvious design decisions
```

## Run it

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # add a free Gemini key from https://aistudio.google.com/apikey
                        # (or set LLM_PROVIDER=openai and OPENAI_API_KEY)

python scripts/smoke_test.py            # check the pipeline, no API key needed (~10 s)
python src/baselines.py                 # baseline numbers, no API key needed
python -m src.eval.calibrate_judge      # LLM judge vs. human scores
python -m src.eval.run_eval --n 20      # evaluate the agent on 20 golden examples
```

Running `run_eval` without `--n` evaluates all 220 examples. That needs a paid key or several days of free-tier quota.

To rebuild `data/pairs.csv` from the raw Kaggle file (`thoughtvector/customer-support-on-twitter`), run:

```bash
python src/data_prep.py --raw data/twcs.csv --brand AppleSupport --out data/pairs.csv --max_pairs 20000
```

## Tech stack

Python · pandas · scikit-learn (TF-IDF, Logistic Regression) · Google Gemini API · OpenAI API (optional) · LLM-as-judge evaluation

---

**Author:** Ajitabh Kumar Jha · [LinkedIn](https://www.linkedin.com/in/ajitabhjha13) · [GitHub](https://github.com/Ajitabhjha13)
