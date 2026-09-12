"""
agent.py
--------
The three-part agent required by the assignment:
  1. classify_intent()   -- one of 9 fixed intents
  2. draft_reply()        -- grounded in retrieved historical resolutions
  3. decide_escalation()  -- auto-handle vs escalate, with a stated reason

Design choice: these are three separate LLM calls, not one mega-prompt.
Rationale (decision_log.md): separating them lets us evaluate and improve
each stage independently (e.g. swap the classifier for a fine-tuned model
later without touching reply generation), and lets escalation logic see the
*already-classified* intent, which measurably simplifies that prompt.
The cost is 3x the API calls per message -- an accepted latency/cost
tradeoff for a support tool where correctness matters more than the
extra ~1-2s.
"""
from __future__ import annotations
import pandas as pd
from src.llm_client import get_client, safe_json_parse
from src.retrieval import TfidfRetriever

INTENTS = [
    "battery_performance", "device_bug", "software_update_issue",
    "account_security", "order_shipping", "billing_refund",
    "howto_info", "praise_feedback", "other_unclear",
]

INTENT_DESCRIPTIONS = """
- battery_performance: battery drain, charging problems, device slowness/heat
- device_bug: a specific malfunction/glitch/crash not clearly tied to a recent update (UI bugs, app crashes, hardware defects, data loss)
- software_update_issue: a problem that started after / is tied to an iOS/macOS/watchOS update
- account_security: login, Apple ID, password reset, 2FA, hacked account, phishing, locked out
- order_shipping: preorders, shipping status, order cancellation/changes
- billing_refund: refund requests, disputed/duplicate charges, subscription billing
- howto_info: a genuine "how do I / does X support Y" informational question, no bug
- praise_feedback: genuine thanks or positive feedback, no unresolved issue
- other_unclear: too little context to classify confidently (fragments, mid-thread replies with no standalone content, off-topic, non-English needing human/translation)
""".strip()

# Intents that structurally require account-specific data or money movement,
# which the historical data shows this brand always moves to a private DM /
# human process for (see report.md "problem framing").
STRUCTURAL_ESCALATE_INTENTS = {"account_security", "billing_refund", "order_shipping", "other_unclear"}


class SupportAgent:
    def __init__(self, pairs_csv: str = "data/pairs.csv"):
        self.pairs = pd.read_csv(pairs_csv)
        self.retriever = TfidfRetriever(self.pairs)
        self.llm = get_client()

    # ---------- 1. Intent classification ----------
    def classify_intent(self, message: str) -> dict:
        system = (
            "You are an intent classifier for Apple's Twitter customer support team. "
            "Classify the customer message into exactly one of these intents:\n\n"
            f"{INTENT_DESCRIPTIONS}\n\n"
            'Respond ONLY as JSON: {"intent": "<one of the 9 labels>", "confidence": <0-1 float>, "reason": "<one short sentence>"}'
        )
        raw = self.llm.complete(system, message, json_mode=True, temperature=0)
        parsed = safe_json_parse(raw)
        if parsed.get("intent") not in INTENTS:
            parsed["intent"] = "other_unclear"
            parsed["_fallback"] = True
        return parsed

    # ---------- 2. Grounded reply drafting ----------
    def draft_reply(self, message: str, intent: str, k: int = 3, exclude_pair_id: str | None = None) -> dict:
        # exclude_pair_id matters for evaluation: golden-set messages are a
        # subsample of pairs.csv, so without excluding the row's own pair_id
        # the retriever would trivially find its own ground-truth answer
        # (similarity=1.0) and "grounding" would be meaningless leakage.
        examples = self.retriever.retrieve(message, k=k, exclude_pair_id=exclude_pair_id)
        examples_block = "\n\n".join(
            f"Similar past case:\nCustomer: {e['customer_message']}\nBrand replied: {e['brand_reply']}"
            for e in examples
        ) or "(no sufficiently similar historical case found)"

        system = (
            "You are drafting a public Twitter reply for Apple's support account, in the "
            "brand's actual historical voice: short, warm, never gives account-specific "
            "detail in public, and moves anything requiring personal/order/account data to DM. "
            "Ground your draft in the style and content of the similar past cases below -- "
            "don't invent a fix that isn't consistent with how this brand has actually replied.\n\n"
            f"Detected intent: {intent}\n\n{examples_block}\n\n"
            'Respond ONLY as JSON: {"draft_reply": "<the tweet-length reply text>", '
            '"grounded_on": "<1 sentence on which past case(s) you drew from, or \'general brand pattern\' if none matched well>"}'
        )
        raw = self.llm.complete(system, message, json_mode=True, temperature=0.4)
        parsed = safe_json_parse(raw)
        parsed["_retrieved_examples"] = examples
        return parsed

    # ---------- 3. Escalation decision ----------
    def decide_escalation(self, message: str, intent: str) -> dict:
        structural_hint = intent in STRUCTURAL_ESCALATE_INTENTS
        system = (
            "You decide whether a customer support agent can auto-handle this Twitter reply "
            "or must escalate to a human agent.\n\n"
            "ESCALATE only if at least one of these concrete conditions holds:\n"
            "  - The message needs account-specific data to resolve (order status, billing/refund, "
            "login/Apple ID recovery, a hacked account).\n"
            "  - The customer is ABUSIVE, uses slurs/threats, or explicitly demands a human/refund/callback.\n"
            "  - The message is non-English.\n"
            "  - There genuinely isn't enough standalone context to safely reply at all (a bare "
            "fragment like 'Nope still doesn't work' with no description of what 'it' is).\n\n"
            "AUTO-HANDLE everything else, INCLUDING:\n"
            "  - A clearly described bug/complaint (battery, crashes, update issues, etc.) even if the "
            "customer sounds annoyed, sarcastic, or uses mild profanity venting about the PRODUCT -- "
            "ordinary frustration about a known issue is not a reason to escalate. Being upset about a "
            "bug is normal; it only escalates if the person themselves is abusive toward the support agent.\n"
            "  - How-to / informational questions.\n"
            "  - Positive feedback.\n"
            "  - A described bug even without an exact known fix -- 'auto-handle' here means 'a human-"
            "reviewable draft reply can be produced', not 'the bug is definitely fixed'.\n\n"
            "Examples:\n"
            "  'Oh god my phone is dying, deader than the dead sea, help' -> auto_handle "
            "(clear battery complaint, sarcastic but not abusive, standard workaround exists)\n"
            "  'And yes, I've restarted my phone, my wifi, my life, your life...' -> auto_handle "
            "(sarcastic venting about repeated troubleshooting, but the issue itself is describable)\n"
            "  'iCloud hacked and I'm locked out of my Mac' -> escalate (account security, needs private "
            "identity verification)\n"
            "  'Nope still doesn't work when I tweet.' -> escalate (no standalone context: doesn't say "
            "what 'it' is)\n\n"
            f"Classified intent: {intent} "
            f"(this intent structurally {'usually needs' if structural_hint else 'usually does not need'} "
            "escalation for this brand historically -- use it as a weak prior only, judge the actual "
            "message on the criteria above).\n\n"
            'Respond ONLY as JSON: {"decision": "auto_handle" | "escalate", "reason": "<one short sentence, specific to this message, citing which criterion above triggered it>"}'
        )
        raw = self.llm.complete(system, message, json_mode=True, temperature=0)
        parsed = safe_json_parse(raw)
        if parsed.get("decision") not in ("auto_handle", "escalate"):
            parsed["decision"] = "escalate"  # fail safe
            parsed["_fallback"] = True
        return parsed

    # ---------- Orchestration ----------
    def handle(self, message: str, exclude_pair_id: str | None = None) -> dict:
        intent_result = self.classify_intent(message)
        intent = intent_result["intent"]
        escalation_result = self.decide_escalation(message, intent)
        reply_result = self.draft_reply(message, intent, exclude_pair_id=exclude_pair_id)
        return {
            "message": message,
            "intent": intent,
            "intent_confidence": intent_result.get("confidence"),
            "escalation_decision": escalation_result["decision"],
            "escalation_reason": escalation_result["reason"],
            "draft_reply": reply_result.get("draft_reply"),
            "grounded_on": reply_result.get("grounded_on"),
        }


if __name__ == "__main__":
    import sys
    agent = SupportAgent()
    msg = sys.argv[1] if len(sys.argv) > 1 else "my iphone battery is draining so fast after the update, please help"
    result = agent.handle(msg)
    for k, v in result.items():
        print(f"{k}: {v}")
