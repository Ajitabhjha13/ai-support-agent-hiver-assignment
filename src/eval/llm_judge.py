"""
llm_judge.py
------------
Scores a drafted reply against a fixed rubric, 1-5 per dimension:

  - grounded:     Does the reply match how this brand has *actually* resolved
                   similar issues historically (not just "sounds plausible")?
  - correct:      Is the content factually/technically sound (no invented
                   steps, no wrong product claims)?
  - brand_voice:  Tone/length/style consistent with the brand's real replies
                   (short, warm, moves account-specific stuff to DM)?
  - safety:       Does it avoid promising things the brand can't deliver
                   (refunds, replacements) and avoid asking for sensitive info
                   in public?

Overall score = mean of the four, rounded to 1 decimal.

This is intentionally a *rubric* judge (fixed criteria, structured JSON output)
rather than a free-text "rate this reply" judge -- free-text judges are harder
to calibrate and to check for human agreement (see calibrate_judge.py).
"""
from src.llm_client import get_client, safe_json_parse

RUBRIC_DIMENSIONS = ["grounded", "correct", "brand_voice", "safety"]


def judge_reply(customer_message: str, draft_reply: str, retrieved_examples: list[dict] | None = None) -> dict:
    llm = get_client()
    examples_block = ""
    if retrieved_examples:
        examples_block = "\n\nHistorical examples the reply was supposed to be grounded on:\n" + "\n".join(
            f"- Customer: {e['customer_message']}\n  Brand replied: {e['brand_reply']}"
            for e in retrieved_examples
        )

    system = f"""You are grading a customer-support reply against a fixed rubric.
Score each dimension 1 (poor) to 5 (excellent):

- grounded: does the reply match how this brand has actually resolved similar issues (per the historical examples), not just sound plausible?
- correct: is the content technically sound, with no invented steps or wrong claims?
- brand_voice: is the tone/length/style consistent with the brand's real replies (short, warm, defers account-specific detail to DM)?
- safety: does it avoid over-promising (refunds/replacements it can't guarantee) and avoid requesting sensitive info in public?
{examples_block}

Respond ONLY as JSON:
{{"grounded": <1-5 int>, "correct": <1-5 int>, "brand_voice": <1-5 int>, "safety": <1-5 int>, "overall": <mean, 1 decimal>, "rationale": "<1-2 sentences>"}}"""

    user = f"Customer message: {customer_message}\n\nDrafted reply: {draft_reply}"
    raw = llm.complete(system, user, json_mode=True, temperature=0)
    parsed = safe_json_parse(raw)
    return parsed


if __name__ == "__main__":
    result = judge_reply(
        "my iphone battery drains so fast after the update",
        "We're here to help! Please DM us your iOS version so we can look into this together.",
        retrieved_examples=[{
            "customer_message": "battery drains fast since ios 11",
            "brand_reply": "We'd love to help. Which iOS version are you using? DM us here.",
        }],
    )
    print(result)
