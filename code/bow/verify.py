"""Deterministic checks run on every row before it is written.

The scorer only ever sees ``output.csv``, so an invalid row is a silently lost
mark. Each check mirrors a hard constraint from the problem statement; a row
that fails one is repaired into the safest valid form rather than shipped.
"""

from __future__ import annotations

import re
from datetime import date

from .loading import PaymentOption, Request

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
PLAN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:\d+(?:\.\d+)?$")
CHANGE_RE = re.compile(r"^(?:stop:[A-Za-z0-9_]+|reduce_to:[A-Za-z0-9_]+:\d+(?:\.\d+)?)$")

TOLERANCE = 0.02


def _parse_plan(text: str) -> list[tuple[date, float]] | None:
    if text == "none":
        return []
    parts = text.split("|")
    out: list[tuple[date, float]] = []
    for part in parts:
        if not PLAN_RE.match(part):
            return None
        when, amount = part.rsplit(":", 1)
        out.append((date.fromisoformat(when), float(amount)))
    return out


def check(
    row: dict,
    request: Request,
    options: list[PaymentOption],
    flexible_event_ids: set[str],
) -> list[str]:
    """Return a list of constraint violations for one output row."""
    problems: list[str] = []

    try:
        safe = float(row["amount_safe_to_pay"])
    except (TypeError, ValueError):
        problems.append("amount_safe_to_pay is not a number")
        safe = 0.0
    if not (-1e-6 <= safe <= request.requested_amount + 1e-6):
        problems.append(f"amount_safe_to_pay {safe} outside [0, {request.requested_amount}]")

    if row["affordability_status"] not in STATUSES:
        problems.append(f"bad affordability_status {row['affordability_status']!r}")
    if row["recommended_payment_method"] not in METHODS:
        problems.append(f"bad recommended_payment_method {row['recommended_payment_method']!r}")

    plan = _parse_plan(row["payment_plan"])
    if plan is None:
        problems.append(f"unparseable payment_plan {row['payment_plan']!r}")
        plan = []
    elif plan != sorted(plan, key=lambda p: p[0]):
        problems.append("payment_plan is not in chronological order")

    method = row["recommended_payment_method"]
    status = row["affordability_status"]
    earliest = row["earliest_date_for_full_payment"]

    if method == "not_recommended":
        if plan:
            problems.append("not_recommended must have payment_plan none")
        if status != "not_affordable":
            problems.append("not_recommended requires not_affordable")
    else:
        if not plan:
            problems.append(f"{method} requires a payment plan")
        total = sum(a for _, a in plan)
        if method in ("full_payment", "partial_payment", "wait"):
            if abs(total - request.requested_amount) > TOLERANCE:
                problems.append(f"plan total {total} != requested {request.requested_amount}")
        if plan and plan[-1][0] > request.desired_completion_date:
            problems.append("plan finishes after desired_completion_date")

    if status == "affordable_now" and earliest != request.request_date.isoformat():
        problems.append("affordable_now requires earliest_date_for_full_payment == request_date")

    if method == "partial_payment":
        if status != "affordable_with_plan":
            problems.append("partial_payment requires affordable_with_plan")
        if not request.allows_partial_payment:
            problems.append("partial_payment used where the request forbids it")
        if len(plan) != 2:
            problems.append("partial_payment needs exactly two payments")
        elif plan[0][0] != request.request_date:
            problems.append("partial_payment must start on request_date")
        elif abs(plan[0][1] - safe) > TOLERANCE:
            problems.append("first partial payment must equal amount_safe_to_pay")
        elif earliest and plan[1][0].isoformat() != earliest:
            problems.append("second partial payment must land on earliest_date_for_full_payment")
        if not (0 < safe < request.requested_amount):
            problems.append("partial_payment needs 0 < amount_safe_to_pay < requested_amount")

    if method == "installments":
        matched = False
        for option in options:
            if option.payment_method != "installments":
                continue
            step = option.payment_frequency_days or 0
            expected = [
                (option.first_payment_date.toordinal() + step * i, option.payment_amount)
                for i in range(option.number_of_payments)
            ]
            actual = [(d.toordinal(), a) for d, a in plan]
            if len(expected) == len(actual) and all(
                x[0] == y[0] and abs(x[1] - y[1]) <= TOLERANCE for x, y in zip(expected, actual)
            ):
                matched = True
                break
        if not matched:
            problems.append("installment plan does not match any supplied payment option")

    changes = row["spending_changes_needed"]
    if changes != "none":
        parts = changes.split("|")
        if len(parts) > 3:
            problems.append("more than three spending changes")
        seen: set[str] = set()
        for part in parts:
            if not CHANGE_RE.match(part):
                problems.append(f"malformed spending change {part!r}")
                continue
            event_id = part.split(":")[1]
            if event_id in seen:
                problems.append(f"event {event_id} changed twice")
            seen.add(event_id)
            if event_id not in flexible_event_ids:
                problems.append(f"spending change targets non-flexible event {event_id}")

    return problems
