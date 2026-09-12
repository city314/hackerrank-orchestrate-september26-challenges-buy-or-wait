"""End-to-end: requests in, validated output rows out."""

from __future__ import annotations

from datetime import date

from . import evidence, explain, forecast, planner, verify
from .loading import Dataset, OUTPUT_COLUMNS, Request


def solve(
    data: Dataset,
    request: Request,
    config: forecast.Config,
    patch: dict | None,
) -> tuple[dict, list[str]]:
    """Produce one output row plus any constraint violations found in it."""
    state = forecast.build(data, request, config, patch)
    options = data.payment_options.get(request.request_id, [])

    safe_today = state.amount_safe_today()
    earliest = state.earliest_full_payment_date()
    plan = planner.choose(state, options)

    # affordable_now is defined by capacity on the request date, so the two
    # fields are derived from the same simulation rather than set separately.
    if plan.status == "affordable_now":
        earliest = request.request_date

    row = {
        "request_id": request.request_id,
        "amount_safe_to_pay": forecast.fmt_amount(safe_today),
        "affordability_status": plan.status,
        "recommended_payment_method": plan.method,
        "payment_plan": plan.render_plan(),
        "earliest_date_for_full_payment": earliest.isoformat() if earliest else "",
        "spending_changes_needed": plan.render_changes(),
        "decision_explanation": explain.compose(state, plan, safe_today),
    }

    flexible = {a.event_id for a in state.adjustable}
    problems = verify.check(row, request, options, flexible)
    if problems:
        row = _fallback(row, request, safe_today, earliest)
        problems = [f"{p} (row replaced with a safe fallback)" for p in problems]
    return row, problems


def _fallback(row: dict, request: Request, safe_today: float, earliest: date | None) -> dict:
    """The most conservative valid row, used when a candidate fails validation."""
    return {
        "request_id": request.request_id,
        "amount_safe_to_pay": forecast.fmt_amount(max(0.0, min(safe_today, request.requested_amount))),
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": row.get("decision_explanation")
        or "The full request cannot be completed safely within the 90-day forecast.",
    }


def run(
    data: Dataset,
    requests: list[Request],
    config: forecast.Config | None = None,
    patches: dict[str, dict] | None = None,
    verbose: bool = True,
) -> tuple[list[dict], list[str]]:
    config = config or forecast.Config()
    patches = patches if patches is not None else evidence.load_cached_patches(data)
    rows, issues = [], []
    for i, request in enumerate(requests, 1):
        row, problems = solve(data, request, config, patches.get(request.request_id))
        rows.append({k: row[k] for k in OUTPUT_COLUMNS})
        issues.extend(f"{request.request_id}: {p}" for p in problems)
        if verbose and i % 50 == 0:
            print(f"  solved {i}/{len(requests)}")
    return rows, issues
