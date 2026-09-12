# AppleSupport AI Agent — Hiver SDE Intern Assignment

An AI customer-support agent for **@AppleSupport**, built on the Kaggle
"Customer Support on Twitter" dataset. Given an incoming customer message it:

1. **Classifies intent** into one of 9 data-derived categories.
2. **Drafts a reply**, grounded in retrieval over how AppleSupport has
   historically resolved similar issues (TF-IDF retrieval, not a black box).
3. **Decides auto-handle vs escalate**, with a stated reason.

See `report.md` for problem framing, results vs. baselines, failure analysis,
and the mandatory "what's misleading about my headline number" section.
See `decision_log.md` for the non-obvious calls made along the way.

## Repo layout

```
src/
  data_prep.py           # raw twcs.csv -> brand-filtered (customer_msg, brand_reply) pairs
  retrieval.py            # TF-IDF grounding retriever
  agent.py                 # the 3-part agent (classify / draft / escalate)
  llm_client.py            # thin OpenAI wrapper
  baselines.py             # trivial + simple (TF-IDF+LogReg) baselines, no LLM needed
  eval/
    build_golden_set.py         # stratified sampling -> draft labels
    finalize_golden_set.py      # human-reviewed corrections -> final golden set
    build_judge_calibration.py  # hand-scored reply-quality calibration set
    calibrate_judge.py          # LLM-judge vs human agreement
    llm_judge.py                 # rubric-based reply quality judge
    run_eval.py                   # full LLM system eval -> headline numbers
data/
  pairs.csv                          # 20k (customer_msg, brand_reply) pairs (reproducible via data_prep.py)
  golden/golden_eval_set.csv         # 220 hand-labeled examples (the golden set)
  golden/golden_labeling_notes.md    # sampling method + correction log
  golden/judge_calibration.csv       # 24 hand-scored reply-quality examples
scripts/
  smoke_test.py             # validates the whole pipeline with a FAKE LLM client (no API key needed)
report.md
decision_log.md
```

## Setup (5 minutes)

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set **one** provider:
- **Free option (no card needed):** get a key at https://aistudio.google.com/apikey,
  set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...`.
- **Paid option:** set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=sk-...` (needs
  billing set up on platform.openai.com; ~$0.35 covers the full 220-example run).

## Reproduce the headline results (under 15 minutes)

**Step 0 — smoke test (no API key / cost needed, ~10 seconds):**
```bash
python scripts/smoke_test.py
```
Confirms retrieval + agent + judge are wired correctly using a fake LLM
client before you spend real API credits.

**Step 1 — regenerate the data pipeline (optional, already committed).**
The repo ships with `data/pairs.csv` already built from a 20k-row subsample
(reproducible, seeded). To rebuild from the raw Kaggle file yourself:
```bash
# download thoughtvector/customer-support-on-twitter from Kaggle, place twcs.csv in data/
python src/data_prep.py --raw data/twcs.csv --brand AppleSupport --out data/pairs.csv --max_pairs 20000
```

**Step 2 — baselines (no API key needed, ~10 seconds):**
```bash
python src/baselines.py
```
Prints trivial-baseline and TF-IDF+LogReg simple-baseline numbers on the
220-example golden set.

**Step 3 — calibrate the LLM judge against human scores (~1 minute, ~$0.01):**
```bash
python -m src.eval.calibrate_judge
```
Reports Pearson correlation + MAE between the judge and our hand-scored
calibration set (`data/golden/judge_calibration.csv`). Check this *before*
trusting any judge-based number below.

**Step 4 — run the full LLM system on a subsample of the golden set
(~2-3 minutes, ~$0.10 for `--n 60`):**
```bash
python -m src.eval.run_eval --n 60
```
Drop `--n 60` to run all 220 examples (~10 minutes, ~$0.35) for the exact
numbers quoted in `report.md`.

## Data note

We do not commit the raw ~500MB `twcs.csv` or a huge derived file. `data/pairs.csv`
(20k pairs, ~9MB) is the reproducible subsample everything else builds on —
this satisfies the assignment's expectation of subsampling rather than
processing the full 3M-row dataset. `--max_pairs` in `data_prep.py` is a
`--seed`-fixed random sample, so re-running it reproduces the same file.
