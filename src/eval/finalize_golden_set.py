"""
finalize_golden_set.py
-----------------------
Applies human-review corrections to data/golden/golden_draft.csv and writes
the final golden evaluation set.

Every row in the draft was read by a human. `OVERRIDES` below is the record
of every correction made and *why* -- this is the audit trail that makes the
"hand-labelled" claim in the assignment defensible. Rows not in OVERRIDES
were reviewed and judged correct as drafted.

Common correction patterns found during review (see report.md, "golden set
sampling & labeling" for the write-up):
  1. Keyword false positives: "order" matched inside "in order for", "out of
     order" (playlist), etc. and got mis-bucketed into order_shipping.
     Similarly "charg" matched "charged" (a billing word) inside battery
     complaints.
  2. Sarcastic "thank you" messages that are actually bug complaints
     ("thank you for screwing up my phone") -- these are NOT praise.
  3. Mid-thread fragments with no standalone context ("Tried that. Same
     problem.", "Just checked, works, on first try.") -- these can't be
     answered without the earlier turns our single-pair extraction dropped.
     We keep them in the set as other_unclear/escalate=yes rather than
     discard them, because a real deployed agent WILL see orphaned replies
     like this and needs a defined (safe) behavior for them.
  4. The recurring "I"-autocorrect bug (iOS 11 replaced typed "I" with the
     wrong glyph) shows up in ~15 different phrasings across the sample --
     a real, single, well-known bug, consistently mapped to
     software_update_issue.
  5. Non-English messages (German, Spanish, Dutch) -- intent still inferable
     from context/keywords, but escalate is forced to "yes": we have no
     reliable automated translation step, so a human should handle these
     rather than risk a wrong-language or wrong-context auto-reply.
"""
import pandas as pd

# index -> (new_intent_or_None, new_escalate_or_None, note)
# None means "keep the draft value for that field"
OVERRIDES = {
    2: ('billing_refund', None, 'charge complaint, not battery (keyword false-matched "charg" in "charged")'),
    23: ('software_update_issue', None, 'explicitly "new ios" - grouped with update-tagged bugs'),
    20: (None, 'yes', 'extreme anger/profanity - escalate for de-escalation despite known bug pattern'),
    27: ('other_unclear', 'yes', 'mid-thread fragment with no standalone complaint'),
    31: (None, 'yes', 'non-English (German) - forcing escalate; translation risk'),
    33: ('howto_info', None, 'Maps data feature request, not an update failure'),
    50: ('device_bug', None, 'iMessage delivery failure - keyword false-matched "delivered"'),
    53: ('device_bug', None, 'music track order bug in app - keyword false-matched "order"'),
    54: ('device_bug', None, 'keyword false-matched "in order for" - actual issue is UI glitch'),
    56: ('software_update_issue', None, 'Safari/High Sierra bug, no real order-keyword basis'),
    62: ('device_bug', None, '"I" autocorrect bug rant; refund/money is venting, not the ask'),
    63: ('device_bug', None, 'camera broken after update; #refund is a hashtag not the core ask'),
    64: ('software_update_issue', None, 'Dutch calendar translation bug after iOS 11'),
    65: ('device_bug', None, '"I" autocorrect bug rant; money mention incidental'),
    70: ('battery_performance', None, 'battery drain complaint; "money" mention is sarcastic aside'),
    72: ('software_update_issue', None, '"I" autocorrect bug'),
    75: ('other_unclear', 'yes', 'mid-thread fragment, no standalone context'),
    78: ('software_update_issue', None, '"I" autocorrect bug'),
    79: ('device_bug', None, 'describes a UI bug (settings screen stuck), not general how-to'),
    85: ('device_bug', None, 'sarcastic complaint about alarm/notification bug, not real praise'),
    87: ('other_unclear', 'yes', 'mid-thread reply providing device model, no standalone context'),
    88: ('device_bug', None, 'sarcastic complaint about an email-related bug'),
    89: ('device_bug', None, 'reports recurring installer error; "thanks" is a thread artifact'),
    90: ('device_bug', None, 'sarcastic complaint about broken laptop + service delay'),
    93: ('device_bug', None, 'reports additional glitches; "thank you" is incidental'),
    95: ('software_update_issue', None, 'sarcastic complaint about update breaking the phone'),
    96: ('device_bug', None, 'Apple Music playback error despite active subscription'),
    97: ('other_unclear', 'yes', 'vague reference to unspecified "eye thing" bug, no context'),
    98: ('device_bug', None, 'Bluetooth disconnect bug'),
    99: ('software_update_issue', None, '"I" autocorrect bug, asked jokingly'),
    100: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    101: ('other_unclear', 'yes', 'non-English (Spanish) + vague WiFi/Bluetooth complaint'),
    102: ('device_bug', None, 'macOS audio balance bug'),
    103: ('device_bug', None, 'fitness badge not awarded despite meeting goal - tracking bug'),
    104: ('other_unclear', 'yes', 'fragment, no clear ask'),
    105: ('other_unclear', 'yes', 'venting with no specific actionable request'),
    106: ('device_bug', 'yes', 'iPod sync failure/hang; also flagged for extreme frustration'),
    107: ('other_unclear', 'yes', '"can you help me?" with zero specifics'),
    110: ('software_update_issue', None, 'slowness/performance regression after update'),
    112: ('other_unclear', 'yes', 'in-store staff complaint, not a digital product issue'),
    113: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    116: ('device_bug', None, 'iPad cannot connect to iTunes to download content'),
    117: ('device_bug', None, "MacBook screen color defect"),
    119: ('other_unclear', 'yes', '"Spinning gear.." - no context'),
    120: ('other_unclear', 'yes', 'complaint about unreachable physical retail store'),
    121: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    122: ('device_bug', None, 'cosmetic display defect (shadow on screen)'),
    126: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    127: ('software_update_issue', None, '"I" autocorrect bug (capital I)'),
    128: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    129: ('software_update_issue', None, 'general frustration with new iOS update'),
    130: ('other_unclear', 'yes', 'complaint about live-chat support quality, not a product issue'),
    132: ('account_security', None, 'error adding card to Apple Pay - payment/account setup issue'),
    133: ('other_unclear', 'yes', 'off-topic/joke message'),
    134: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    136: ('device_bug', None, 'video editing app performance complaint'),
    137: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    138: ('software_update_issue', None, 'explicit iOS11 bug complaint'),
    139: ('other_unclear', 'yes', 'reservation status reverted, context-free'),
    140: ('other_unclear', 'yes', 'no specifics, needs clarification'),
    141: ('other_unclear', 'yes', 'context-free fragment'),
    143: ('other_unclear', 'yes', '"Help me" - zero specifics'),
    144: ('device_bug', None, 'watch feature regression, continuing question not praise'),
    145: ('device_bug', None, 'contacts showing as numbers not names in iMessage'),
    147: ('other_unclear', 'yes', 'fragment referencing "I" bug without context'),
    148: ('device_bug', None, 'lost contacts + missed calls, functional bug'),
    150: ('howto_info', None, 'question about iCloud sign-out procedure'),
    151: ('other_unclear', 'yes', 'no content'),
    152: ('software_update_issue', None, '"I" autocorrect bug (emoji conflation)'),
    153: ('device_bug', None, 'Apple TV 4K disappointment/return'),
    154: ('billing_refund', None, 'Apple Music monthly charge question'),
    155: ('howto_info', None, 'AirPods + Apple TV compatibility question'),
    156: ('device_bug', None, 'keyboard bug; "parents paid money" is incidental'),
    157: ('other_unclear', 'yes', 'vague frustration, no specific ask'),
    160: ('other_unclear', 'yes', 'issue self-resolved, no action needed'),
    164: ('other_unclear', 'yes', 'context-free fragment'),
    165: ('device_bug', None, 'Game Center stuck loading'),
    166: ('device_bug', None, '"I" bug, annoyed'),
    167: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    170: ('other_unclear', 'yes', 'no specifics'),
    171: ('software_update_issue', None, '"I" autocorrect bug'),
    172: ('other_unclear', 'yes', 'context-free fragment'),
    173: ('device_bug', None, 'phone performance complaint after new release'),
    175: ('device_bug', None, 'flashlight control greyed out, UI bug'),
    176: ('software_update_issue', None, 'cannot complete update, "thanks" is incidental'),
    179: ('device_bug', None, 'alarm app malfunction'),
    180: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    182: ('howto_info', None, 'feature suggestion, not a bug'),
    183: ('device_bug', None, 'photos missing after iCloud restore - data loss bug'),
    184: ('howto_info', None, 'charger compatibility question'),
    185: ('device_bug', None, 'new phone setup/box issue'),
    186: ('other_unclear', 'yes', '"I have that problem too" lacks specifics'),
    187: ('device_bug', None, 'activity tracker inaccurate tracking bug'),
    188: ('other_unclear', 'yes', 'vague, no specifics'),
    189: ('other_unclear', 'yes', 'bare version number, no context'),
    193: ('device_bug', None, 'phone overheating'),
    194: ('software_update_issue', None, 'random auto-capitalization bug'),
    195: ('device_bug', None, 'apps not closing during call - iPhone X bug'),
    196: ('battery_performance', None, 'phone dying/battery complaint'),
    197: ('software_update_issue', 'yes', 'iOS 11.0.3 bug (Spanish) - non-English, force escalate'),
    198: ('order_shipping', None, 'in-store reserve-and-pickup complaint'),
    199: ('other_unclear', 'yes', 'context-free mid-thread reply'),
    200: ('software_update_issue', None, '"I" bug, ongoing frustration'),
    202: ('howto_info', None, 'question about generating app-specific passwords'),
    203: ('device_bug', None, 'WiFi/Bluetooth toggling itself on unexpectedly'),
    206: ('other_unclear', 'yes', 'prior help/callback not honored - needs human follow-up'),
    207: ('software_update_issue', None, '"I" bug (# and ? substitution)'),
    208: ('device_bug', None, 'iPhone X activation failure'),
    209: ('software_update_issue', None, '"I" bug (question mark box)'),
    210: ('software_update_issue', None, '"I" bug (cannot type letter I)'),
    211: ('other_unclear', 'yes', 'bare device/version info, no complaint'),
    214: ('other_unclear', 'yes', 'ambiguous confusing fragment'),
    215: ('device_bug', None, 'Watch cannot stream music from phone after WatchOS change'),
    216: ('howto_info', None, 'cable/charging compatibility question'),
    217: ('device_bug', None, 'notification center display bug'),
    218: ('device_bug', None, 'status bar UI stuck/blocking bug'),
}

ESCALATE_INTENTS = {"account_security", "billing_refund", "order_shipping", "other_unclear"}


def finalize(draft_path: str, out_path: str, notes_path: str):
    df = pd.read_csv(draft_path).reset_index(drop=True)
    final_intent = df["draft_intent"].tolist()
    final_escalate = df["draft_escalate"].tolist()
    note_col = [""] * len(df)

    for idx, (intent_ov, esc_ov, note) in OVERRIDES.items():
        if intent_ov is not None:
            final_intent[idx] = intent_ov
        if esc_ov is not None:
            final_escalate[idx] = esc_ov
        elif intent_ov is not None:
            # intent changed but escalate not explicitly overridden -> recompute
            final_escalate[idx] = "yes" if final_intent[idx] in ESCALATE_INTENTS else "no"
        note_col[idx] = note

    df["intent"] = final_intent
    df["escalate"] = final_escalate
    df["correction_note"] = note_col
    df["was_corrected"] = df.index.isin(OVERRIDES.keys())

    final = df[[
        "pair_id", "customer_message", "brand_reply", "intent", "escalate",
        "was_corrected", "correction_note",
    ]]
    final.to_csv(out_path, index=False)

    n_corrected = df["was_corrected"].sum()
    with open(notes_path, "w") as f:
        f.write(f"# Golden Set Labeling Notes\n\n")
        f.write(f"Total examples: {len(df)}\n")
        f.write(f"Corrected on human review: {n_corrected} ({100*n_corrected/len(df):.1f}%)\n\n")
        f.write("## Final intent distribution\n\n")
        f.write(final["intent"].value_counts().to_string())
        f.write("\n\n## Final escalate distribution\n\n")
        f.write(final["escalate"].value_counts().to_string())
        f.write("\n\n## Sample corrections (draft -> final)\n\n")
        sample = df[df["was_corrected"]].head(15)
        for _, row in sample.iterrows():
            f.write(f"- \"{row['customer_message'][:90]}\"\n")
            f.write(f"    draft={row['draft_intent']} -> final={row['intent']}  "
                    f"[{row['correction_note']}]\n")

    print(f"Final golden set: {len(final)} rows -> {out_path}")
    print(f"Corrected: {n_corrected} ({100*n_corrected/len(df):.1f}%)")
    print(final["intent"].value_counts())
    print()
    print(final["escalate"].value_counts())


if __name__ == "__main__":
    finalize(
        "data/golden/golden_draft.csv",
        "data/golden/golden_eval_set.csv",
        "data/golden/golden_labeling_notes.md",
    )
