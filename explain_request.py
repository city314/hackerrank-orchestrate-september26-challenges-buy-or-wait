"""Explain one request end to end.

    python explain_request.py request_137

Prints why the engine answered as it did: the user's profile and preferences,
any evidence patch applied, the recurring series that were detected, the
projected timeline up to the trough, and the final row. Useful for debugging a
single case without reading the whole forecast by hand.

A development tool - not part of the submitted code package.
"""
from __future__ import annotations
import sys, csv, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
from bow import evidence, forecast, planner                     # noqa: E402
from bow.loading import Dataset                                 # noqa: E402

rid = sys.argv[1] if len(sys.argv) > 1 else "request_26"
data = Dataset()
req = next((r for r in data.requests + data.sample_requests if r.request_id == rid), None)
if req is None:
    raise SystemExit(f"no such request: {rid}")

p = data.profiles[req.user_id]
patch = evidence.load_cached_patches(data).get(rid)
f = forecast.build(data, req, forecast.Config(), patch)
plan = planner.choose(f, data.payment_options.get(rid, []))

row = {}
if os.path.exists("output.csv"):
    with open("output.csv", encoding="utf-8") as fh:
        row = next((r for r in csv.DictReader(fh) if r["request_id"] == rid), {})

print(f"\n{rid}  ({req.user_id}, {p.home_currency})  {req.request_type}")
print(f"  asked        {req.requested_amount:,.2f} by {req.desired_completion_date}"
      f"   partial allowed: {req.allows_partial_payment}")
print(f"  balance      {p.current_available_balance:,.2f}   minimum {p.minimum_balance_to_keep:,.2f}")
print(f"  accepts      {'|'.join(p.payment_methods)}   max installments: {p.max_installment_months}")
print(f"  protects     {'|'.join(p.protect) or '-'}")
print(f"  will reduce  {'|'.join(p.willing_to_reduce) or '-'}   will stop: {'|'.join(p.willing_to_stop) or '-'}")

if patch and any(patch[k] for k in ("event_amounts", "cancelled_event_ids",
                                    "event_date_changes", "income_update",
                                    "recurring_expense_changes")):
    print("\n  EVIDENCE APPLIED")
    for k in ("event_amounts", "cancelled_event_ids", "event_date_changes",
              "income_update", "recurring_expense_changes"):
        if patch[k]:
            print(f"    {k}: {patch[k]}")
    for n in patch.get("notes", []):
        print(f"    note: {n}")
else:
    print("\n  EVIDENCE: none applied" + (" (message present but nothing actionable)"
                                          if patch else ""))

print("\n  RECURRING SERIES")
for s in f.series:
    cadence = f"day {s.day_of_month}" if s.day_of_month else f"every {s.period_days}d"
    print(f"    {s.category:20s} {cadence:12s} n={len(s.occurrences):2d} "
          f"-> {s.forecast_amount(f.config.variable_policy):12,.2f}  {s.flexibility}")
inc = [x for x in f.committed if x.amount > 0]
print(f"    income projected: {len(inc)} payments"
      + (f", first {inc[0].on} of {inc[0].amount:,.2f}" if inc else " - NONE"))

print("\n  TIMELINE (to the trough)")
bal, low, lowd = f.start_balance, f.start_balance, req.request_date
for fl in f.base_flows():
    bal += fl.amount
    if bal < low:
        low, lowd = bal, fl.on
for fl in f.base_flows():
    if fl.on > lowd:
        break
    print(f"    {fl.on}  {fl.label:30s} {fl.amount:>14,.2f}")
print(f"    trough {low:,.2f} on {lowd}  -  minimum {p.minimum_balance_to_keep:,.2f}"
      f"  =>  safe {low - p.minimum_balance_to_keep:,.2f} (capped at requested)")

print(f"\n  ANSWER   {row.get('affordability_status', plan.status)} / "
      f"{row.get('recommended_payment_method', plan.method)}")
print(f"    safe     {row.get('amount_safe_to_pay', '')}")
print(f"    plan     {row.get('payment_plan', plan.render_plan())}")
print(f"    earliest {row.get('earliest_date_for_full_payment', '') or '(none)'}")
print(f"    changes  {row.get('spending_changes_needed', plan.render_changes())}")
print(f"    says     {row.get('decision_explanation', '')}\n")
