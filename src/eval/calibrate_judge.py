"""
calibrate_judge.py
-------------------
Runs the LLM judge (src/eval/llm_judge.py) on every row of
data/golden/judge_calibration.csv and reports:
  - Pearson correlation (judge overall vs human overall)
  - Mean absolute error (judge overall vs human overall)
  - Exact-match-within-1-point rate (a looser, more interpretable agreement metric)
  - Per-dimension correlation (grounded/correct/brand_voice/safety)

Requires OPENAI_API_KEY (see README) -- this is the one part of the eval
harness that cannot be verified without an LLM API call. Run this first,
before trusting any judge scores elsewhere in the pipeline: if agreement is
weak here, the reply-quality numbers downstream should not be taken at face
value (see report.md, "what is misleading about my headline number").
"""
import pandas as pd
import numpy as np
from src.eval.llm_judge import judge_reply


def calibrate(calibration_path: str = "data/golden/judge_calibration.csv"):
    df = pd.read_csv(calibration_path)
    judge_overalls, judge_grounded, judge_correct, judge_voice, judge_safety = [], [], [], [], []

    for _, row in df.iterrows():
        result = judge_reply(
            row["customer_message"], row["reply"],
            retrieved_examples=[{
                "customer_message": row["customer_message"],
                "brand_reply": row["reference_reply"],
            }],
        )
        judge_overalls.append(result.get("overall", np.nan))
        judge_grounded.append(result.get("grounded", np.nan))
        judge_correct.append(result.get("correct", np.nan))
        judge_voice.append(result.get("brand_voice", np.nan))
        judge_safety.append(result.get("safety", np.nan))

    df["judge_overall"] = judge_overalls
    df["judge_grounded"] = judge_grounded
    df["judge_correct"] = judge_correct
    df["judge_brand_voice"] = judge_voice
    df["judge_safety"] = judge_safety
    df.to_csv("data/golden/judge_calibration_results.csv", index=False)

    def pearson(a, b):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 2:
            return float("nan")
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    mae = float(np.nanmean(np.abs(df["judge_overall"] - df["human_overall"])))
    within_1 = float((np.abs(df["judge_overall"] - df["human_overall"]) <= 1.0).mean())

    print(f"N = {len(df)}")
    print(f"Overall score  -- Pearson r: {pearson(df['judge_overall'], df['human_overall']):.3f} | "
          f"MAE: {mae:.3f} | within 1 point: {within_1:.1%}")
    print(f"grounded       -- Pearson r: {pearson(df['judge_grounded'], df['human_grounded']):.3f}")
    print(f"correct        -- Pearson r: {pearson(df['judge_correct'], df['human_correct']):.3f}")
    print(f"brand_voice    -- Pearson r: {pearson(df['judge_brand_voice'], df['human_brand_voice']):.3f}")
    print(f"safety         -- Pearson r: {pearson(df['judge_safety'], df['human_safety']):.3f}")
    print("\nSaved per-row comparison -> data/golden/judge_calibration_results.csv")


if __name__ == "__main__":
    calibrate()
