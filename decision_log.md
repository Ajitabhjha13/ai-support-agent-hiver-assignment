# Decision Log

Non-obvious calls made building this, and why.

1. **Unit of work = first customer message → first brand reply, not full
   threads.** Real threads are 3-6 messages deep. Reconstructing full threads
   correctly (handling branching replies, multiple customers in one thread,
   etc.) was judged to cost more engineering time than it would add
   evaluation signal for a first version. Named explicitly as the top "what
   I'd do next" item so it doesn't read as an oversight.

2. **TF-IDF retrieval instead of neural embeddings.** No external model
   download needed (works in network-restricted grading environments), and
   Twitter support text is short/keyword-heavy where lexical retrieval is
   already competitive. Swappable in ~10 lines (`EmbeddingRetriever` stub in
   `src/retrieval.py`).

3. **9 intents, not more.** The assignment asks for "a small set." We started
   from keyword-frequency exploration of the raw data (battery, update/bug,
   account, order, spam, howto, praise, app-specific ≈ 8 rough buckets),
   then added `other_unclear` as an explicit catch-all rather than forcing
   ambiguous messages into a wrong specific bucket — better to have an honest
   catch-all than an artificially high-confidence wrong label.

4. **Escalation ground truth derived partly from the brand's own historical
   behavior**, not invented from scratch: AppleSupport's real replies show a
   consistent pattern (DM redirect for anything account/order/billing-
   specific, public workaround for known bugs). We used this as a strong
   prior for which intents "structurally" require escalation, then had a
   human (not the rule) make the final call per message — see `escalate`
   column derivation in `finalize_golden_set.py`.

5. **Rule-based draft labels before human review, not human-labeling from a
   blank slate.** Every one of the 220 golden-set rows was read and judged by
   a human, but starting from a cheap heuristic draft (regex keyword rules)
   made the review faster. The tradeoff is documented honestly: 52.7% of
   draft labels were wrong and got corrected — the rules were a speed aid,
   not a reliable labeler on their own (see `golden_labeling_notes.md`).

6. **Kept "praise_feedback" as its own intent despite being rare (~2% of
   final labels)**, rather than folding it into `other_unclear`. Genuine
   thanks needs a genuinely different (much shorter, no escalation) handling
   path than an unclear/ambiguous message, even though both are low-volume.

7. **Sarcastic "thank you" messages are labeled by their real content, not
   their surface politeness.** *"Thank you for screwing up my phone"* is
   `device_bug`/`software_update_issue`, not `praise_feedback`. This
   surfaced during labeling review (9 cases) and is called out as Failure
   Mode #3 in the report because it's a genuinely hard case for any
   classifier, human-labeled or not.

8. **Non-English messages always escalate, regardless of what intent they'd
   otherwise get.** We found German, Dutch, and Spanish examples in a random
   English-brand sample. Rather than attempt automatic translation (out of
   scope, and a wrong translation could produce a confidently-wrong reply),
   these are forced to `escalate=yes` in the golden labels and in the
   agent's escalation prompt.

9. **`exclude_pair_id` in the retriever exists specifically to prevent
   evaluation leakage**, because the golden set is a subsample of the same
   `pairs.csv` corpus the retriever indexes. Without it, retrieval on a
   golden-set query trivially finds that exact historical answer
   (similarity 1.0), and "grounded reply quality" would be measuring nothing.
   Caught during development (see report.md Section 4) rather than baked in
   from the start — kept in the log as a reminder for anyone extending
   the golden set later.

10. **Golden set is stratified-then-topped-up, not pure random.** Pure random
    sampling from raw traffic would be ~35% software-update-bug/battery
    examples and could statistically miss rarer classes like
    `billing_refund` (which is <1% of raw traffic) entirely. We stratified
    against rough keyword buckets first, then filled the remaining budget
    randomly to still catch intents the keywords miss (this is literally how
    `other_unclear` was discovered as a large, real category rather than
    invented on paper).

11. **Two baselines, not one, and both are 100% reproducible offline.**
    The trivial baseline (majority class) sets the absolute floor. The simple
    baseline (TF-IDF + Logistic Regression via 5-fold cross-validation on
    the golden set itself, since we have no larger labeled training set) is
    what "a competent engineer's afternoon, no LLM" looks like. Both run in
    seconds with no API key, so a grader can verify them even before setting
    up an LLM key.

12. **Reply generation, intent classification, and escalation are three
    separate LLM calls, not one combined prompt.** Costs 3x the calls, but
    lets each stage be evaluated, debugged, and improved independently (e.g.
    the escalation prompt can see the classified intent as context, which
    measurably simplifies it vs. re-deriving intent signals from scratch).

13. **The LLM-judge is a fixed 4-dimension rubric (grounded / correct /
    brand_voice / safety) with structured JSON output, not a free-text "rate
    this reply 1-10."** Structured rubrics are dramatically easier to
    calibrate against human scores (see `calibrate_judge.py`) — a free-text
    judge would require parsing/normalizing scores before any agreement
    metric was even possible.

14. **Judge calibration examples were constructed (not randomly sampled)
    to deliberately span the quality spectrum** — actual historical replies
    (expected ~5), generic ungrounded replies (~2.5-3), unsafe replies that
    over-promise or request sensitive info (~1), and well-drafted grounded
    replies (~4.5-5). Random sampling of real (message, reply) pairs would
    have given almost all "good" examples (since most are the brand's real,
    reasonable replies) and produced no signal on whether the judge can tell
    good from bad.

15. **Provider-agnostic LLM client, with Gemini as the free-tier default.**
    `src/llm_client.py` supports OpenAI, Google Gemini, and has an Anthropic
    stub. Gemini's free tier (AI Studio) needs no billing/card, which matters
    for a take-home assignment being reproduced by a grader who may not want
    to add a payment method just to run someone else's code — this was
    switched to the default after initially building against OpenAI only.
