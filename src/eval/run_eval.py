"""
run_eval.py
-----------
The "headline results" script referenced in README.md. Runs the full LLM
agent (classify -> escalate -> draft reply) over every row in the golden
eval set, scores replies with the LLM judge, and prints/saves a metrics
table comparing against both baselines (src/baselines.py).

Usage:
    python -m src.eval.run_eval --n 60      # subsample for a fast run
    python -m src.eval.run_eval             # full 220-example golden set

Requires OPENAI_API_KEY. Each row costs ~3 LLM calls (classify, escalate,
draft) + 1 judge call = 4 calls; budget accordingly (--n lets you control cost
for a quick sanity check before running the full set).
"""
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.agent import SupportAgent
from src.eval.llm_judge import judge_reply


def run(golden_path: str, pairs_path: str, out_path: str, n: int | None = None, seed: int = 42):
    df = pd.read_csv(golden_path)
    if n:
        df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    agent = SupportAgent(pairs_csv=pairs_path)

    records = []
    for i, row in df.iterrows():
        msg = row["customer_message"]
        intent_result = agent.classify_intent(msg)
        pred_intent = intent_result["intent"]
        escalation_result = agent.decide_escalation(msg, pred_intent)
        # exclude this row's own pair_id -- golden set is a subsample of
        # pairs.csv, so without this the retriever leaks the ground-truth
        # reply back to itself (see agent.py docstring on exclude_pair_id)
        reply_result = agent.draft_reply(msg, pred_intent, exclude_pair_id=row["pair_id"])

        judge_result = judge_reply(
            msg, reply_result.get("draft_reply", ""),
            retrieved_examples=reply_result.get("_retrieved_examples"),
        )

        records.append({
            "pair_id": row["pair_id"],
            "customer_message": msg,
            "gold_intent": row["intent"],
            "pred_intent": pred_intent,
            "gold_escalate": row["escalate"],
            "pred_escalate": "yes" if escalation_result["decision"] == "escalate" else "no",
            "escalation_reason": escalation_result["reason"],
            "draft_reply": reply_result.get("draft_reply"),
            "grounded_on": reply_result.get("grounded_on"),
            "judge_overall": judge_result.get("overall"),
            "judge_grounded": judge_result.get("grounded"),
            "judge_correct": judge_result.get("correct"),
            "judge_brand_voice": judge_result.get("brand_voice"),
            "judge_safety": judge_result.get("safety"),
        })
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(df)} done")

    results = pd.DataFrame(records)
    results.to_csv(out_path, index=False)

    intent_acc = accuracy_score(results["gold_intent"], results["pred_intent"])
    intent_f1 = f1_score(results["gold_intent"], results["pred_intent"], average="macro")
    escalate_acc = accuracy_score(results["gold_escalate"], results["pred_escalate"])
    escalate_f1 = f1_score(results["gold_escalate"], results["pred_escalate"], pos_label="yes")
    mean_judge = results["judge_overall"].mean()

    print("\n=== LLM SYSTEM (headline results) ===")
    print(f"N = {len(results)}")
    print(f"Intent accuracy: {intent_acc:.3f} | macro-F1: {intent_f1:.3f}")
    print(f"Escalation accuracy: {escalate_acc:.3f} | F1 (escalate=yes): {escalate_f1:.3f}")
    print(f"Mean judged reply quality (1-5): {mean_judge:.2f}")
    print(f"\nFull per-row results -> {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden/golden_eval_set.csv")
    ap.add_argument("--pairs", default="data/pairs.csv")
    ap.add_argument("--out", default="data/golden/llm_eval_results.csv")
    ap.add_argument("--n", type=int, default=None, help="subsample size (omit for full set)")
    args = ap.parse_args()
    run(args.golden, args.pairs, args.out, n=args.n)
