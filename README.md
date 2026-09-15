# Buy or Wait?

An affordability agent that decides whether someone can safely pay for something they
want, and if not, what they should do instead.

Built for [HackerRank Orchestrate](https://www.hackerrank.com/hackerrank-orchestrate-september26),
a 24 hour AI engineering hackathon. **Ranked 352 of 3,062 participants.**

Given a request such as *"Can I afford this laptop?"*, the system answers with a
number and a plan: how much is safe to pay today, and whether to pay in full, pay
part now, use one of the offered installment options, wait, or not proceed at all.

---

## The idea

**Keep the language model out of the arithmetic.**

Four of the six scored fields are exact values: an amount, a date, a payment
schedule. Language models drift on arithmetic and do not reproduce it run to run.
So the numbers come from a deterministic simulator, and the model is confined to the
one part of the problem that is genuinely linguistic: reading untrusted messages and
payslip photographs.

```
dataset/  ->  reconstruct state  ->  90 day forecast  ->  rank plans  ->  verify  ->  output.csv
                      ^
               evidence patch (LLM)
```

`amount_safe_to_pay` is the lowest balance the 90 day forecast reaches, minus the
minimum the user wants to keep, capped at what they asked for. Every candidate plan
is tested against that same simulation, so the recommendation can never contradict
the figures printed beside it.

The model returns a fixed schema with no field an injected instruction could occupy,
and every value is revalidated before the forecast sees it. Event ids must exist,
amounts must be plausible, dates must parse, categories must come from a known set.

---

## Run it

```bash
python code/main.py               # full run: extract evidence, then solve
python code/main.py --no-llm      # deterministic only, no API calls
python code/main.py --samples     # solve the 25 solved examples and self score
python code/evaluation/main.py    # score every field, with the error distribution
python explain_request.py request_137   # why one request answered the way it did
```

Python 3.10+ and pandas. Nothing else. The model is called over plain HTTPS with the
standard library. For the evidence layer, put a Gemini key in `.env` at the root:

```
GOOGLE_API_KEY=your-key-here
```

Without a key the pipeline still runs end to end and writes a complete, valid
`output.csv`. It simply skips the message and image evidence.

---

## What it cost

| | |
|---|---|
| Model | `gemini-3.5-flash-lite`, JSON mode, temperature 0 |
| Model calls | 42 |
| Tokens | 411,902 in, 17,343 out |
| Cost | $0.048 for the whole dataset |
| Evidence patches | 219 |
| Failed calls | 0 |

219 requests carry a message or an image. The free tier allows 20 requests per day on
the default flash model, so one call per request could never finish. Requests without
images are batched eight to a call, and only the sixteen carrying an image are sent
alone. Responses are cached by a hash of the payload, so a rerun costs nothing.

---

## Five decisions worth reading the code for

**Income is separated by what the dataset calls it, not by whether the amount is
stable.** The `salary` category mixes streams. A payroll credit and a performance
commission both arrive monthly, but the spec forbids counting commission until it
settles, while a second household salary moves month to month and is entirely real.
Amount stability gets this wrong in both directions. The sharpest case: one user's
final credit is described as *"Final employer payroll"*. The job had ended.
Projecting that salary forward was a 2000% error on that request.

**A one off purchase must not join a recurring series.** A bulk grocery bill of
INR 41,272 sat inside a weekly shop averaging 8,600, and because it landed on the
same date as a real occurrence an early version summed them into a single 49,988
instalment and projected it forward. Series now drop amount outliers and occurrences
off the cadence grid.

**Two parameters that only work together.** A recurring bill landing exactly on the
request date was being dropped, though its latest recorded occurrence is strictly
earlier so it cannot have been paid. Counting it alone took the sample score *down*
from 63% to 55%, because the spend estimate was already over reserving. Relaxing that
estimate alone did little either. Together they took 62% to 71.3%.

**A policy that scored higher and was rejected.** Taking the minimum of the last
three occurrences as the spend forecast beat the shipped mean of the last four by
about a point. It was dropped for being optimistic by construction: a backtest across
all 275 users, trained on earlier history and tested on weeks it had not seen, showed
it understates spending for 68% of them, against 33% for the policy kept. An
optimistic bias approves requests the user cannot afford, which is the error the whole
task exists to prevent.

**Several messages are deliberate distractors.** A prize "still processing", a refund
"initiated but not credited", a transfer between the user's own accounts. They have no
matching transaction at all. Their purpose is to punish an agent that invents income
from hopeful wording. The engine only acts on event ids that exist, so they correctly
do nothing.

[`code/README.md`](./code/README.md) covers the module layout and the reasoning in
more depth.

---

## Results, including what went wrong

| Component | Score |
|---|---|
| Code package | 22.2 / 30 |
| AI judge interview | 21.3 / 30 |
| Output predictions | 14.1 / 30 |
| Chat transcript | 2.7 / 10 |
| **Final** | **60.3 / 100** |

Two things worth being straight about.

The self scored 71.3% on the 25 solved examples was a *training* score. Those same 25
rows were used to calibrate the forecast parameters, and the hidden set came in well
below it. The categorical fields held up better than the amount, which is what you
would expect when the amount depends on hitting an exact value and the rest depend on
comparisons.

The transcript score is a procedural miss rather than a technical one. The brief asked
for a chat transcript and what I submitted was a structured progress log: summarised
prose, one entry per working session rather than per turn. It satisfied the format I
had inferred instead of the deliverable that was actually asked for.

---

## Layout

```
code/
  main.py                 entry point
  bow/
    loading.py            typed dataset access
    fx.py                 dated currency conversion
    recurrence.py         cadence detection, outlier and off grid rejection
    income.py             separates income that can be counted from income that cannot
    forecast.py           90 day simulation, amount_safe_to_pay, earliest safe date
    planner.py            candidate plans and the six way ranking rule
    verify.py             hard constraint checks with a conservative fallback
    schema.py             the evidence patch contract and its validator
    evidence.py           prompt construction, batching, extraction
    llm.py                Gemini REST client, disk cache, token accounting
    explain.py            decision_explanation, generated from the committed numbers
    scoring.py            self scoring against the solved examples
  evaluation/
    main.py               scoring and output validation entry point
    usage_report.md       token and cost report for the final run
dataset/                  provided input data, unmodified
output.csv                the submitted predictions
explain_request.py        per request debugging tool
```

---

## The challenge

Full task specification in [`problem_statement.md`](./problem_statement.md). The
dataset in `dataset/` is provided by the organisers and is unmodified; the starter
repository is
[interviewstreet/hackerrank-orchestrate-september26](https://github.com/interviewstreet/hackerrank-orchestrate-september26).

For every row in `dataset/requests.csv` the solution writes one row to `output.csv`:

```
request_id, amount_safe_to_pay, affordability_status, recommended_payment_method,
payment_plan, earliest_date_for_full_payment, spending_changes_needed,
decision_explanation
```

A recommendation counts as safe only if the user can complete every listed payment,
finish by the deadline, cover essential spending, and stay above their preferred
minimum balance for the whole forecast period.
