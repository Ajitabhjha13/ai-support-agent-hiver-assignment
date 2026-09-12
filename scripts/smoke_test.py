"""
smoke_test.py
-------------
Sanity-checks that the whole pipeline is wired correctly (retrieval ->
agent -> judge) WITHOUT calling a real LLM API, using a deterministic fake
client. Run this first if you want to confirm the code runs before spending
API credits on the real run_eval.py.

    python scripts/smoke_test.py
"""
import sys
import json
sys.path.insert(0, ".")

import src.llm_client as llm_client


class FakeClient:
    """Returns plausible-looking JSON so downstream parsing code gets exercised."""
    def complete(self, system, user, json_mode=False, temperature=0.3, max_tokens=500):
        if "intent classifier" in system:
            return json.dumps({"intent": "battery_performance", "confidence": 0.8,
                                "reason": "mentions battery drain"})
        if "escalation" in system.lower() or "escalate" in system.lower():
            return json.dumps({"decision": "auto_handle",
                                "reason": "known battery issue with public workaround"})
        if "drafting a public Twitter reply" in system:
            return json.dumps({"draft_reply": "We hear you! Please check out our battery tips here, "
                                               "and DM us if it persists.",
                                "grounded_on": "similar battery drain cases in history"})
        if "grading a customer-support reply" in system:
            return json.dumps({"grounded": 4, "correct": 4, "brand_voice": 5, "safety": 5,
                                "overall": 4.5, "rationale": "matches historical pattern"})
        return json.dumps({})


llm_client.get_client = lambda: FakeClient()

from src.agent import SupportAgent
from src.eval.llm_judge import judge_reply

print("1. Loading retriever + agent (TF-IDF, no API needed for this part)...")
agent = SupportAgent(pairs_csv="data/pairs.csv")
print("   OK -- retriever indexed", len(agent.pairs), "historical pairs")

print("\n2. Running agent.handle() with a fake LLM client...")
result = agent.handle("my iphone battery drains so fast since the update")
for k, v in result.items():
    print(f"   {k}: {v}")

print("\n3. Running the judge with a fake LLM client...")
judge_result = judge_reply(
    "my iphone battery drains so fast since the update",
    result["draft_reply"],
    retrieved_examples=None,
)
print("   ", judge_result)

print("\nSMOKE TEST PASSED -- pipeline is wired correctly.")
print("Next: set GEMINI_API_KEY (free tier) or OPENAI_API_KEY in .env, then run:")
print("  python -m src.eval.run_eval --n 20")
