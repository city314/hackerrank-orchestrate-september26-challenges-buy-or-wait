"""Score generated rows against the 25 solved examples.

The hidden ground truth is scored on seven fields, so the self-check reports
each of them separately: a run can look healthy on ``amount_safe_to_pay`` while
quietly getting every recommendation wrong.
"""

from __future__ import annotations

from .loading import Dataset

FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]
# Relative tolerance for treating a money value as correct.
MONEY_TOLERANCE = 0.005


def _match(field: str, got, want) -> bool:
    if field == "amount_safe_to_pay":
        try:
            got_f, want_f = float(got), float(want)
        except (TypeError, ValueError):
            return False
        return abs(got_f - want_f) <= max(0.01, MONEY_TOLERANCE * abs(want_f))
    return str(got).strip() == str(want).strip()


def report(data: Dataset, rows: list[dict], show_misses: bool = True) -> dict[str, float]:
    answers = data.sample_answers
    scores = {f: 0 for f in FIELDS}
    misses: list[str] = []
    counted = 0

    for row in rows:
        rid = row["request_id"]
        if rid not in answers.index:
            continue
        counted += 1
        want_row = answers.loc[rid]
        wrong = []
        for field in FIELDS:
            want = want_row[field]
            want = "" if str(want) == "nan" else want
            if _match(field, row[field], want):
                scores[field] += 1
            else:
                wrong.append(f"{field}: got {row[field]!r} want {want!r}")
        if wrong:
            misses.append(f"{rid}\n      " + "\n      ".join(wrong))

    print(f"\n=== self-score on {counted} solved samples ===")
    for field in FIELDS:
        pct = 100 * scores[field] / max(counted, 1)
        bar = "#" * int(pct / 5)
        print(f"  {field:32s} {scores[field]:3d}/{counted}  {pct:5.1f}%  {bar}")
    overall = sum(scores.values()) / max(counted * len(FIELDS), 1)
    print(f"  {'OVERALL':32s} {100 * overall:5.1f}%")

    if show_misses and misses:
        print(f"\n  {len(misses)} row(s) with at least one mismatch:")
        for line in misses:
            print(f"    {line}")
    return {f: scores[f] / max(counted, 1) for f in FIELDS}
