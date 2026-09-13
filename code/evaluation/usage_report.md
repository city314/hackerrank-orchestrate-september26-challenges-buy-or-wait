# Token Usage and Cost Analysis

Final full-dataset run that produced `output.csv`.
Generated 2026-09-13 10:45 UTC.

## Model

| Field | Value |
|---|---|
| Provider | Google |
| Model | `gemini-3.5-flash-lite` |
| Interface | REST `generateContent`, JSON mode, temperature 0 |
| Requests in run | 250 |

Only one model is used. The affordability arithmetic, plan ranking and
validation are deterministic Python and make no model calls; the model is used
solely to turn messages and images into a schema-checked evidence patch.

## Calls and tokens

| Metric | Value |
|---|---|
| Model calls | 42 |
| — served from cache | 42 |
| — sent to the API | 0 |
| Failed calls | 0 |
| Input tokens | 411,902 |
| Output tokens | 17,343 |
| **Total tokens** | **429,245** |
| Average tokens per request | 1,717.0 |

Requests whose user has no message and no image make no call at all, so the
call count is below the request count by design.

## Cost

| Metric | Value |
|---|---|
| Estimated total cost | $0.0481 |
| Estimated cost per request | $0.000193 |

Costed at the published `gemini-3.5-flash-lite` rate. Responses are cached on disk by
a hash of the request payload, so re-running the pipeline reproduces
`output.csv` at zero additional cost.
