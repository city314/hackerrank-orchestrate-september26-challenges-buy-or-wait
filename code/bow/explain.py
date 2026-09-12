"""Compose ``decision_explanation`` in the house style of the solved samples.

The samples are tightly formulaic, and the scorer rewards usefulness *and*
consistency, so the wording is generated deterministically from the numbers the
planner already committed to. That also guarantees the sentence can never
contradict the row it sits on.
"""

from __future__ import annotations

from datetime import date

from .forecast import Forecast
from .planner import Plan

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def money(currency: str, value: float) -> str:
    """Format like the samples: thousands separators, decimals only if needed."""
    rounded = round(float(value), 2)
    if abs(rounded - round(rounded)) < 0.005:
        return f"{currency} {int(round(rounded)):,}"
    return f"{currency} {rounded:,.2f}"


def long_date(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def _describe(event_id: str, descriptions: dict[str, str]) -> str:
    text = descriptions.get(event_id, "").strip()
    return text[0].lower() + text[1:] if text else "this recurring expense"


def _changes_clause(plan: Plan, currency: str, descriptions: dict[str, str]) -> str:
    parts = []
    for adjustment in plan.adjustments:
        label = _describe(adjustment.event_id, descriptions)
        if adjustment.kind == "stop":
            parts.append(f"stop the {label}")
        else:
            parts.append(f"reduce the {label} to {money(currency, adjustment.new_amount)}")
    if not parts:
        return ""
    if len(parts) == 1:
        clause = parts[0]
    else:
        clause = ", ".join(parts[:-1]) + " and " + parts[-1]
    return clause[0].upper() + clause[1:]


def compose(forecast: Forecast, plan: Plan, safe_today: float) -> str:
    request = forecast.request
    currency = forecast.profile.home_currency
    floor = money(currency, forecast.profile.minimum_balance_to_keep)
    descriptions = forecast.descriptions

    if plan.method == "full_payment":
        amount = money(currency, plan.payments[0][1])
        clause = _changes_clause(plan, currency, descriptions)
        if clause:
            return f"{clause}, then pay {amount} today. This leaves at least {floor} available."
        return (
            f"Pay {amount} today. This leaves at least {floor} available "
            f"over the next 90 days."
        )

    if plan.method == "installments":
        count = len(plan.payments)
        each = money(currency, plan.payments[0][1])
        start = long_date(plan.payments[0][0])
        clause = _changes_clause(plan, currency, descriptions)
        lead = f"{clause}, then use" if clause else "Use"
        return (
            f"{lead} {count} installments of {each}, starting {start}. "
            f"This leaves at least {floor} available."
        )

    if plan.method == "partial_payment":
        first = money(currency, plan.payments[0][1])
        second = money(currency, plan.payments[1][1])
        when = long_date(plan.payments[1][0])
        return (
            f"Pay {first} today and the remaining {second} on {when}. "
            f"This completes the full request and keeps the {floor} minimum protected."
        )

    if plan.method == "wait":
        amount = money(currency, request.requested_amount)
        when = long_date(plan.payments[0][0])
        return (
            f"Pay {amount} in full on {when}. Paying earlier would take the balance "
            f"below the {floor} minimum."
        )

    # not_recommended
    if safe_today > 0:
        return (
            f"Do not proceed with the {money(currency, request.requested_amount)} request. "
            f"Although {money(currency, safe_today)} is available today, the full amount "
            f"cannot be completed safely within 90 days."
        )
    return (
        f"Do not make this payment by {long_date(request.desired_completion_date)}. "
        f"None of the available options keeps the {floor} minimum protected."
    )
