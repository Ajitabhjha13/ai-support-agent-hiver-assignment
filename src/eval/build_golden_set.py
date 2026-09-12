"""
build_golden_set.py
--------------------
Builds the candidate pool for our 150-250 example golden evaluation set.

Sampling method (documented for report.md):
    1. Stratified sampling across 8 keyword-defined rough buckets (so rare
       intents like Billing/Order aren't drowned out by the ~35% iOS-bug
       majority class), plus an "unstratified random" pool to catch intents
       our keywords miss entirely (this is exactly how "Other/Unclear"
       examples get discovered).
    2. A cheap rule-based function proposes a DRAFT intent + escalation label.
    3. A human (us) reviews and corrects every single row -- the draft is a
       productivity aid, not the ground truth. See golden_labeling_notes.md
       for what changed between draft and final, and why.

This keeps labeling honest: an LLM-assisted first pass is fine, but the
"hand-labelled" claim only holds if a human actually looked at every row,
which we did (see the ~18% of rows in golden_labeling_notes.md that had
their draft label overturned on review).
"""
import argparse
import re
import pandas as pd

INTENTS = [
    "battery_performance",
    "device_bug",
    "software_update_issue",
    "account_security",
    "order_shipping",
    "billing_refund",
    "howto_info",
    "praise_feedback",
    "other_unclear",
]

RULES = [
    ("account_security", r"sign.?in|log.?in|password|verif|2fa|two.factor|apple\s?id|locked out|hack|spam|scam|phish"),
    ("billing_refund", r"refund|charged twice|overcharg|billing|subscription.*cancel|money"),
    ("order_shipping", r"order|ship|deliver|preorder|tracking number|package"),
    ("praise_feedback", r"^(thank|thanks|love|great|awesome|appreciate)"),
    ("battery_performance", r"battery|charg(e|ing)|drain"),
    ("software_update_issue", r"ios 1[0-9]|update|beta|upgrad"),
    ("device_bug", r"crash|freez|restart|glitch|broken|doesn.?t work|stopped working|won.?t (open|turn|start)"),
    ("howto_info", r"^how (do|to|can)|how do i|what is|does .* support|can i"),
]


def draft_label(text: str) -> tuple[str, str]:
    t = text.lower()
    intent = "other_unclear"
    for label, pattern in RULES:
        if re.search(pattern, t):
            intent = label
            break
    escalate_intents = {"account_security", "billing_refund", "order_shipping", "other_unclear"}
    escalate = "yes" if intent in escalate_intents else "no"
    return intent, escalate


def sample_pool(pairs_path: str, n: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(pairs_path)
    df["draft_intent"], df["draft_escalate"] = zip(*df["customer_message"].map(draft_label))

    # stratified: take a roughly even slice per bucket, then top up randomly
    per_bucket = max(1, n // (2 * len(INTENTS)))  # half the budget stratified
    strat_parts = []
    for intent in INTENTS:
        pool = df[df["draft_intent"] == intent]
        strat_parts.append(pool.sample(n=min(per_bucket, len(pool)), random_state=seed))
    strat = pd.concat(strat_parts)

    remaining = n - len(strat)
    rest_pool = df.drop(strat.index)
    rest = rest_pool.sample(n=min(remaining, len(rest_pool)), random_state=seed)

    out = pd.concat([strat, rest]).drop_duplicates(subset="pair_id").reset_index(drop=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs.csv")
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--out", default="data/golden/golden_draft.csv")
    args = ap.parse_args()

    pool = sample_pool(args.pairs, args.n)
    pool.to_csv(args.out, index=False)
    print(f"Sampled {len(pool)} candidates -> {args.out}")
    print(pool["draft_intent"].value_counts())
