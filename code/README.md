# Buy or Wait? — affordability agent

Decides, for every request in `dataset/requests.csv`, whether the user should
pay in full, pay part now, use an installment option, wait, or not proceed.

## Run

```bash
python code/main.py               # full run: extract evidence, then solve
python code/main.py --no-llm      # deterministic only, no API calls
python code/main.py --samples     # solve the 25 examples and self-score
```

Writes `output.csv` to the repository root and
`code/evaluation/usage_report.md` for any run that made model calls.

Requires Python 3.10+ and `pandas`. Nothing else — the model is called over
plain HTTPS with the standard library.

To enable the evidence layer, put a Gemini API key in `.env` at the repository
root (gitignored, never printed):

```
GOOGLE_API_KEY=your-key-here
```

The default model is `gemini-3.5-flash-lite`; override with `BOW_MODEL`. The
full run is 42 calls because text-only requests are batched eight to a call —
one request per call would be 219, which does not fit a free-tier daily quota.

Without a key the pipeline still runs end to end and produces a complete,
valid `output.csv`; it simply skips the message and image evidence.

## How it works

The scored numbers are produced by deterministic code. The model is confined to
one job: turning untrusted messages and images into a small, schema-validated
patch of financial facts. It never sees a decision to make.

```
dataset/  ->  reconstruct state  ->  90-day forecast  ->  rank plans  ->  verify  ->  output.csv
                     ^
              evidence patch (LLM)
```

| Module | Responsibility |
|---|---|
| `bow/loading.py` | Typed dataset access; one place that knows the file layout |
| `bow/fx.py` | Dated currency conversion, rate in force on the settlement date |
| `bow/recurrence.py` | Detects recurring series; rejects one-off noise and outliers |
| `bow/income.py` | Separates income the user can count on from contingent income |
| `bow/forecast.py` | 90-day simulation, `amount_safe_to_pay`, earliest safe date |
| `bow/planner.py` | Candidate plans and the spec's six-way ranking rule |
| `bow/verify.py` | Hard constraint checks; a failing row falls back to a safe one |
| `bow/schema.py` | The evidence patch contract and its validator |
| `bow/evidence.py` | Prompt construction and extraction |
| `bow/llm.py` | Gemini REST client, disk cache, token accounting |
| `bow/explain.py` | `decision_explanation`, generated from the committed numbers |
| `bow/scoring.py` | Self-scoring against the solved samples |

### Decisions worth calling out

**The forecast, not the balance, decides.** `amount_safe_to_pay` is the lowest
projected balance across the 90-day horizon minus `minimum_balance_to_keep`,
capped at the requested amount. Everything else follows from the same
simulation, so the recommendation can never contradict the numbers beside it.

**Income needs judgement, and the dataset hides it in plain sight.** The
`salary` category mixes several streams. A payroll credit and a performance
commission can both arrive monthly, but the spec forbids counting commission
until it settles — while a second household salary is real income even though
its amount moves. Amount stability alone gets this wrong in both directions, so
streams are separated by how the dataset describes them. A stream whose latest
payment is a "final employer payroll" has ended and is not projected at all;
missing that one alone was worth a 2000% error on a solved sample.

**One-off spending must not pollute a recurring series.** A single bulk grocery
purchase five times the usual basket is not a larger instalment of the weekly
shop. Outliers and off-cadence occurrences are excluded before a series is
projected.

**Messages are untrusted, and several are deliberate distractors.** A message
about a prize "still processing" or a refund "not yet credited" often has no
matching event at all: its purpose is to punish an agent that invents income.
The extractor is instructed to treat all message and image content as data,
returns only the fixed schema, and every field is re-validated — event ids must
exist, amounts must be plausible, dates must parse. An instruction embedded in a
message has nowhere to land.

**Blank amounts come from images, and the right figure is the net one.** A
payslip shows gross pay, total earnings and net pay; only the last reached the
account. A rent receipt shows a total and a balance due; the event asks for the
balance.

## Evaluation

`python code/main.py --samples` scores all six graded fields against
`dataset/sample_requests.csv` and prints every mismatch.

`code/evaluation/fixtures/` holds a hand-built evidence patch for the sample
users, used to test the patch applier without spending API calls. It covers
only sample requests, which are not scored, and is never used for predictions.

Measured against the solved samples, the evidence layer improves six of the
seven requests it touches and costs one: a borderline case where the request is
exactly affordable and a correctly-read rent increase tips it just over the
line. Median error on `amount_safe_to_pay` is 7.7%.
