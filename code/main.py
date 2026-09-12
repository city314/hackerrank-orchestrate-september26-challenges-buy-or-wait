"""Buy or Wait? — entry point.

    python code/main.py                 # full run: extract evidence, then solve
    python code/main.py --no-llm        # deterministic only, no API calls
    python code/main.py --samples       # score against the 25 solved examples

Writes ``output.csv`` to the repository root.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bow import evidence, forecast, llm, pipeline, usage
from bow.loading import OUTPUT_COLUMNS, REPO_ROOT, Dataset


def load_dotenv(path: str = os.path.join(REPO_ROOT, ".env")) -> None:
    """Read KEY=VALUE lines from .env without adding a dependency.

    Secrets stay out of the repository: .env is gitignored and nothing here
    ever prints a value.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? affordability agent")
    parser.add_argument("--no-llm", action="store_true", help="skip evidence extraction")
    parser.add_argument("--samples", action="store_true", help="run the 25 solved examples")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "output.csv"))
    parser.add_argument("--model", default=llm.DEFAULT_MODEL)
    args = parser.parse_args()

    load_dotenv()
    data = Dataset()
    requests = data.sample_requests if args.samples else data.requests
    print(f"Buy or Wait? — {len(requests)} requests")

    client = llm.Client(model=args.model)
    if args.no_llm:
        patches = evidence.load_cached_patches(data)
        print(f"  evidence: LLM disabled, {len(patches)} cached patches reused")
    else:
        if not client.available:
            print("  evidence: no GOOGLE_API_KEY set — falling back to cached patches")
            patches = evidence.load_cached_patches(data)
        else:
            print(f"  evidence: extracting with {args.model}")
            # Evidence is per user, so both request sets share one cache.
            patches = {
                **evidence.load(),
                **evidence.extract(data, requests, client),
            }
            evidence.save(patches)
            known = set(data.events["event_id"])
            from bow import schema

            patches = {rid: schema.validate(p, known) for rid, p in patches.items()}
            print(f"  evidence: {len(patches)} patches ready")

    rows, issues = pipeline.run(data, requests, forecast.Config(), patches)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")

    if issues:
        print(f"\n{len(issues)} validation issue(s):")
        for line in issues[:20]:
            print(f"  - {line}")
    else:
        print("All rows passed validation.")

    if client.usage.calls:
        usage.write_report(client, len(requests))
        print(f"Usage report written to {usage.REPORT_PATH}")

    if args.samples:
        from bow import scoring

        scoring.report(data, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
