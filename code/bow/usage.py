"""Write ``evaluation/usage_report.md`` for the run that produced output.csv."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from .llm import Client

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
REPORT_PATH = os.path.join(EVAL_DIR, "usage_report.md")


def write_report(client: Client, request_count: int, path: str = REPORT_PATH) -> str:
    stats = client.report(request_count)
    live = stats["calls"] - stats["cached_calls"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    body = f"""# Token Usage and Cost Analysis

Final full-dataset run that produced `output.csv`.
Generated {generated}.

## Model

| Field | Value |
|---|---|
| Provider | {stats['provider']} |
| Model | `{stats['model']}` |
| Interface | REST `generateContent`, JSON mode, temperature 0 |
| Requests in run | {request_count} |

Only one model is used. The affordability arithmetic, plan ranking and
validation are deterministic Python and make no model calls; the model is used
solely to turn messages and images into a schema-checked evidence patch.

## Calls and tokens

| Metric | Value |
|---|---|
| Model calls | {stats['calls']:,} |
| — served from cache | {stats['cached_calls']:,} |
| — sent to the API | {live:,} |
| Failed calls | {stats['failures']:,} |
| Input tokens | {stats['input_tokens']:,} |
| Output tokens | {stats['output_tokens']:,} |
| **Total tokens** | **{stats['total_tokens']:,}** |
| Average tokens per request | {stats['avg_tokens_per_request']:,} |

Requests whose user has no message and no image make no call at all, so the
call count is below the request count by design.

## Cost

| Metric | Value |
|---|---|
| Estimated total cost | ${stats['estimated_cost_usd']:.4f} |
| Estimated cost per request | ${stats['estimated_cost_per_request_usd']:.6f} |

Costed at the published `{stats['model']}` rate. Responses are cached on disk by
a hash of the request payload, so re-running the pipeline reproduces
`output.csv` at zero additional cost.
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path
