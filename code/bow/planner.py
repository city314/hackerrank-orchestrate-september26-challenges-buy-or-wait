"""Generate candidate plans and pick the one the spec ranks first.

Every candidate is checked against the same 90-day safety simulation, so the
recommendation and the numbers reported alongside it can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations

from .forecast import Adjustment, Forecast, _fmt
from .loading import PaymentOption

MAX_SPENDING_CHANGES = 3


@dataclass
class Plan:
    """One safe way to proceed, ready to be ranked against the others."""

    method: str
    status: str
    payments: tuple[tuple[date, float], ...]
    adjustments: tuple[Adjustment, ...] = ()
    option: PaymentOption | None = None

    @property
    def total_paid(self) -> float:
        return sum(amount for _, amount in self.payments)

    @property
    def completes_on(self) -> date | None:
        return max((on for on, _ in self.payments), default=None)

    def render_plan(self) -> str:
        if not self.payments:
            return "none"
        return "|".join(f"{on.isoformat()}:{_fmt(amount)}" for on, amount in self.payments)

    def render_changes(self) -> str:
        if not self.adjustments:
            return "none"
        return "|".join(a.render() for a in self.adjustments)

    def rank_key(self, deadline: date) -> tuple:
        """The documented preference order, lowest tuple wins."""
        completes = self.completes_on
        return (
            0 if (completes is not None and completes <= deadline) else 1,  # 1. by deadline
            len(self.adjustments),                                          # 2. no changes
            round(self.total_paid, 2),                                      # 3. cheapest
            self.payments[0][0] if self.payments else date.max,             # 4. start earlier
            len(self.payments),                                             # 5. fewer payments
            self.option.sort_key if self.option else -1,                    # 6. lowest option id
        )


def _installment_dates(option: PaymentOption) -> list[date]:
    step = option.payment_frequency_days or 0
    return [
        option.first_payment_date + timedelta(days=step * i)
        for i in range(option.number_of_payments)
    ]


def _spending_change_sets(forecast: Forecast) -> list[tuple[Adjustment, ...]]:
    """Permitted combinations of spending changes, smallest first.

    Stopping and reducing the same event are mutually exclusive, so a
    combination never names one event twice.
    """
    options = sorted(forecast.adjustable, key=lambda a: (-a.saving_per_occurrence, a.event_id))
    out: list[tuple[Adjustment, ...]] = []
    for size in range(1, MAX_SPENDING_CHANGES + 1):
        for combo in combinations(options, size):
            if len({a.event_id for a in combo}) != size:
                continue
            out.append(combo)
    return out


def candidates(forecast: Forecast, options: list[PaymentOption]) -> list[Plan]:
    """Every safe plan the user's preferences allow."""
    request = forecast.request
    profile = forecast.profile
    rd, full = request.request_date, request.requested_amount
    plans: list[Plan] = []

    safe_today = forecast.amount_safe_today()
    earliest = forecast.earliest_full_payment_date()

    # --- pay in full today ---------------------------------------------------
    if profile.accepts("full_payment"):
        if forecast.is_safe(((rd, full),)):
            plans.append(Plan("full_payment", "affordable_now", ((rd, full),)))
        else:
            # Permitted spending changes can make today's full payment safe.
            for combo in _spending_change_sets(forecast):
                if forecast.is_safe(((rd, full),), combo):
                    plans.append(
                        Plan("full_payment", "affordable_with_plan", ((rd, full),), combo)
                    )
                    break

    # --- installments --------------------------------------------------------
    if profile.accepts("installments"):
        cap = profile.max_installment_months
        for option in options:
            if option.payment_method != "installments":
                continue
            if cap is not None and option.number_of_payments > cap:
                continue
            schedule = tuple((on, option.payment_amount) for on in _installment_dates(option))
            if forecast.is_safe(schedule):
                plans.append(
                    Plan("installments", "affordable_with_plan", schedule, (), option)
                )
                continue
            for combo in _spending_change_sets(forecast):
                if forecast.is_safe(schedule, combo):
                    plans.append(
                        Plan("installments", "affordable_with_plan", schedule, combo, option)
                    )
                    break

    # --- pay part today, the rest on the first safe date ---------------------
    if (
        profile.accepts("partial_payment")
        and request.allows_partial_payment
        and earliest is not None
        and earliest <= request.desired_completion_date
        and 0 < safe_today < full
    ):
        rest = round(full - safe_today, 2)
        schedule = ((rd, safe_today), (earliest, rest))
        if forecast.is_safe(schedule):
            plans.append(Plan("partial_payment", "affordable_with_plan", schedule))

    # --- wait for the full amount to become safe -----------------------------
    if (
        profile.accepts("full_payment")
        and earliest is not None
        and earliest > rd
        and earliest <= request.desired_completion_date
    ):
        plans.append(Plan("wait", "affordable_later", ((earliest, full),)))

    return plans


def choose(forecast: Forecast, options: list[PaymentOption]) -> Plan:
    """Pick the best safe plan, or fall back to recommending against it."""
    plans = [
        p
        for p in candidates(forecast, options)
        if p.completes_on is not None
        and p.completes_on <= forecast.request.desired_completion_date
    ]
    if not plans:
        return Plan("not_recommended", "not_affordable", ())
    return min(plans, key=lambda p: p.rank_key(forecast.request.desired_completion_date))
