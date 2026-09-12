"""
data_prep.py
------------
Filters the raw Kaggle "Customer Support on Twitter" dataset (twcs.csv) down to
one brand, and reconstructs (customer_message -> brand_reply) conversation
pairs -- the atomic unit our agent is trained/evaluated on.

Usage:
    python src/data_prep.py --raw data/twcs.csv --brand AppleSupport \
        --out data/pairs.csv --max_pairs 20000

Why we build "pairs" instead of using raw rows directly:
    The raw dataset is one row per tweet, linked by response_tweet_id /
    in_response_to_tweet_id. A single customer complaint can span multiple
    tweets, and a single brand reply is often the *second* or *third* message
    in a thread (after clarifying questions). For this assignment we take the
    simplifying, clearly-stated decision (see decision_log.md) to model a
    conversation pair as exactly:
        first customer inbound tweet  -> first brand outbound reply to it
    We deliberately ignore deeper multi-turn threads (see report.md,
    "what we chose not to build").
"""
import argparse
import re
import pandas as pd


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r"https?://\S+", "", t)          # strip links
    t = re.sub(r"@\w+", "", t)                   # strip @mentions (incl. brand handle)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_pairs(raw_path: str, brand: str, max_pairs: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["author_id"] = df["author_id"].astype(str)

    brand_replies = df[df["author_id"] == brand].copy()
    brand_replies = brand_replies.dropna(subset=["in_response_to_tweet_id"])

    customer_tweets = df[df["inbound"] == True].set_index("tweet_id")  # noqa: E712

    rows = []
    for _, reply in brand_replies.iterrows():
        cust_id = reply["in_response_to_tweet_id"]
        if pd.isna(cust_id) or int(cust_id) not in customer_tweets.index:
            continue
        cust_row = customer_tweets.loc[int(cust_id)]
        # a tweet_id can rarely map to >1 row after dedup edge cases; guard it
        if isinstance(cust_row, pd.DataFrame):
            cust_row = cust_row.iloc[0]

        cust_text = clean_text(cust_row["text"])
        reply_text = clean_text(reply["text"])
        if len(cust_text) < 3 or len(reply_text) < 3:
            continue

        rows.append({
            "pair_id": f"{int(cust_id)}_{int(reply['tweet_id'])}",
            "customer_tweet_id": int(cust_id),
            "brand_reply_tweet_id": int(reply["tweet_id"]),
            "customer_message": cust_text,
            "brand_reply": reply_text,
            "customer_created_at": cust_row["created_at"],
            "reply_created_at": reply["created_at"],
        })

    out = pd.DataFrame(rows).drop_duplicates(subset="pair_id")
    out = out.sort_values("customer_tweet_id").reset_index(drop=True)
    if max_pairs:
        out = out.sample(n=min(max_pairs, len(out)), random_state=42).sort_values(
            "customer_tweet_id"
        ).reset_index(drop=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/twcs.csv", help="path to raw Kaggle twcs.csv")
    ap.add_argument("--brand", default="AppleSupport")
    ap.add_argument("--out", default="data/pairs.csv")
    ap.add_argument("--max_pairs", type=int, default=20000,
                     help="subsample size for a fast, reproducible run (assignment "
                          "explicitly expects a subsample, not the full 3M rows)")
    args = ap.parse_args()

    pairs = build_pairs(args.raw, args.brand, args.max_pairs)
    pairs.to_csv(args.out, index=False)
    print(f"Built {len(pairs)} (customer_message -> brand_reply) pairs for {args.brand}")
    print(f"Saved to {args.out}")
