"""
baselines.py
------------
Two baselines the assignment requires us to beat:

1. TRIVIAL baseline: always predict the majority class. For escalation,
   always predict the majority action. This is the floor -- if our system
   can't beat "always guess device_bug / always auto-handle", it's not
   adding value.

2. SIMPLE baseline: TF-IDF + Logistic Regression, trained and evaluated with
   5-fold cross-validation *on the golden set itself* (we deliberately do not
   have a separate large labeled training set -- see decision_log.md). This
   represents "a competent engineer's afternoon" solution: no LLM calls, no
   prompt engineering, just classic ML on hand-labeled data.

Both run fully offline -- no API key required -- so a grader can reproduce
these numbers even without an LLM key configured.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report


def trivial_baseline(golden_path: str):
    df = pd.read_csv(golden_path)
    maj_intent = df["intent"].mode()[0]
    maj_escalate = df["escalate"].mode()[0]

    intent_acc = (df["intent"] == maj_intent).mean()
    escalate_acc = (df["escalate"] == maj_escalate).mean()

    print("=== TRIVIAL BASELINE ===")
    print(f"Always predict intent='{maj_intent}': accuracy = {intent_acc:.3f}")
    print(f"Always predict escalate='{maj_escalate}': accuracy = {escalate_acc:.3f}")
    return {"intent_acc": intent_acc, "escalate_acc": escalate_acc,
            "maj_intent": maj_intent, "maj_escalate": maj_escalate}


def simple_baseline(golden_path: str, seed: int = 42):
    df = pd.read_csv(golden_path)
    X_text = df["customer_message"].fillna("")
    y_intent = df["intent"]
    y_escalate = df["escalate"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # --- Intent classifier ---
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, stop_words="english")
    X = vec.fit_transform(X_text)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    intent_preds = cross_val_predict(clf, X, y_intent, cv=skf)
    intent_acc = accuracy_score(y_intent, intent_preds)
    intent_f1 = f1_score(y_intent, intent_preds, average="macro")

    print("\n=== SIMPLE BASELINE: TF-IDF + Logistic Regression (5-fold CV) ===")
    print(f"Intent accuracy: {intent_acc:.3f} | macro-F1: {intent_f1:.3f}")
    print(classification_report(y_intent, intent_preds, zero_division=0))

    # --- Escalation classifier (binary) ---
    clf2 = LogisticRegression(max_iter=1000, class_weight="balanced")
    escalate_preds = cross_val_predict(clf2, X, y_escalate, cv=skf)
    escalate_acc = accuracy_score(y_escalate, escalate_preds)
    escalate_f1 = f1_score(y_escalate, escalate_preds, pos_label="yes")

    print(f"Escalation accuracy: {escalate_acc:.3f} | F1 (escalate=yes): {escalate_f1:.3f}")
    print(classification_report(y_escalate, escalate_preds, zero_division=0))

    return {
        "intent_acc": intent_acc, "intent_macro_f1": intent_f1,
        "escalate_acc": escalate_acc, "escalate_f1": escalate_f1,
    }


if __name__ == "__main__":
    trivial_baseline("data/golden/golden_eval_set.csv")
    simple_baseline("data/golden/golden_eval_set.csv")
