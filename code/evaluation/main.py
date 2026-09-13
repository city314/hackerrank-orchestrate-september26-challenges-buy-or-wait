"""Evaluation entry point.

    python code/evaluation/main.py            # score the solved samples
    python code/evaluation/main.py --check    # validate an existing output.csv

Scores every graded field separately against `dataset/sample_requests.csv`,
because a run can look healthy on `amount_safe_to_pay` while getting the
recommendations wrong. `--check` re-runs the deterministic verifier over a
generated `output.csv` so an invalid row cannot reach the scorer unnoticed.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

from bow import evidence, forecast, pipeline, scoring, verify  # noqa: E402
from bow.loading import Dataset  # noqa: E402


def score_samples(data: Dataset) -> None:
    patches = evidence.load_cached_patches(data)
    rows, issues = pipeline.run(data, data.sample_requests, forecast.Config(), patches, False)
    if issues:
        print(f"{len(issues)} validation issue(s):")
        for line in issues:
            print(f"  - {line}")
    scoring.report(data, rows)

    answers = data.sample_answers
    errors = [
        abs(float(r["amount_safe_to_pay"]) - float(answers.loc[r["request_id"], "amount_safe_to_pay"]))
        / max(float(answers.loc[r["request_id"], "amount_safe_to_pay"]), 1.0)
        for r in rows
    ]
    within = lambda t: sum(1 for e in errors if e <= t)  # noqa: E731
    print(
        f"\n  amount_safe_to_pay relative error: median {statistics.median(errors):.1%} | "
        f"within 5%: {within(0.05)}/{len(errors)} | "
        f"10%: {within(0.10)}/{len(errors)} | 25%: {within(0.25)}/{len(errors)}"
    )


def check_output(data: Dataset, path: str) -> int:
    """Re-validate a generated output.csv against the hard constraints."""
    if not os.path.exists(path):
        print(f"missing {path}")
        return 1
    with open(path, encoding="utf-8") as fh:
        rows = {r["request_id"]: r for r in csv.DictReader(fh)}

    expected = {r.request_id: r for r in data.requests}
    problems: list[str] = []
    for rid, request in expected.items():
        row = rows.get(rid)
        if row is None:
            problems.append(f"{rid}: missing from output")
            continue
        state = forecast.build(data, request, forecast.Config(), None)
        flexible = {a.event_id for a in state.adjustable}
        options = data.payment_options.get(rid, [])
        problems.extend(f"{rid}: {p}" for p in verify.check(row, request, options, flexible))
    for extra in set(rows) - set(expected):
        problems.append(f"{extra}: not a request in requests.csv")

    print(f"{len(rows)} rows checked against {len(expected)} requests")
    if problems:
        print(f"{len(problems)} problem(s):")
        for line in problems[:40]:
            print(f"  - {line}")
        return 1
    print("all rows valid")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? evaluation")
    parser.add_argument("--check", action="store_true", help="validate output.csv instead")
    parser.add_argument("--out", default=os.path.join(REPO, "output.csv"))
    args = parser.parse_args()

    data = Dataset()
    if args.check:
        return check_output(data, args.out)
    score_samples(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
