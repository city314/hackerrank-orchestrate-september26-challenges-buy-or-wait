"""Reconstruct a user's financial state and forecast it forward.

Everything the scorer looks at (``amount_safe_to_pay``,
``earliest_date_for_full_payment``, plan feasibility) comes out of the
``Forecast`` object built here, so this module is deliberately explicit about
which records move cash and when.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from . import income, recurrence
from .fx import Converter
from .loading import DEAD_STATUSES, Dataset, Profile, Request

HORIZON_DAYS = 90


@dataclass(frozen=True)
class Config:
    """Knobs for the parts of the spec that admit more than one reading.

    Defaults are the ones calibrated against ``sample_requests.csv``.
    """

    horizon_days: int = HORIZON_DAYS
    # How to project the next amount of a variable recurring series.
    variable_policy: str = "max3"
    # How to project a variable (gig-economy) income stream.
    income_policy: str = "mean"
    # Project recurring income forward from settled history.
    project_income: bool = True
    # Continue a scheduled income beyond its stated date on the same cadence.
    extend_scheduled_income: bool = True
    # Pending debits are reserved; pending credits are never counted.
    reserve_pending_debits: bool = True
    count_scheduled_credits: bool = True


@dataclass
class Flow:
    """A single dated cash movement in the forecast."""

    on: date
    amount: float  # signed: positive is money in
    label: str
    event_id: str | None = None
    category: str | None = None
    source: str = "recurring"  # recurring | committed | evidence | plan


@dataclass
class Adjustment:
    """A spending change the user is willing to make."""

    event_id: str
    kind: str  # "stop" or "reduce_to"
    new_amount: float | None
    category: str
    saving_per_occurrence: float

    def render(self) -> str:
        if self.kind == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{_fmt(self.new_amount)}"


def _fmt(value: float) -> str:
    """Render a plan or spending-change amount the way the dataset does.

    Whole amounts lose the decimal point entirely (``25256``); fractional ones
    keep exactly two places, trailing zero included (``620.40``).
    """
    rounded = round(float(value) + 0.0, 2)
    if abs(rounded - round(rounded)) < 5e-3:
        return str(int(round(rounded)))
    return f"{rounded:.2f}"


def fmt_amount(value: float) -> str:
    """Render ``amount_safe_to_pay``: shortest exact form, no trailing zeros."""
    rounded = round(float(value) + 0.0, 2)
    if abs(rounded - round(rounded)) < 5e-3:
        return str(int(round(rounded)))
    return f"{rounded:.2f}".rstrip("0")


@dataclass
class Forecast:
    """A user's reconstructed position plus the machinery to test a plan."""

    profile: Profile
    request: Request
    series: list[recurrence.Series]
    committed: list[Flow]
    config: Config
    start_balance: float
    horizon_end: date
    adjustable: list[Adjustment] = field(default_factory=list)
    expense_changes: list[dict] = field(default_factory=list)
    # event_id -> human description, used when wording a spending change
    descriptions: dict[str, str] = field(default_factory=dict)

    # ---- timeline construction -------------------------------------------------

    def base_flows(self, adjustments: tuple[Adjustment, ...] = ()) -> list[Flow]:
        """Every projected movement in the horizon, before any request payment."""
        stopped = {a.event_id for a in adjustments if a.kind == "stop"}
        reduced = {a.event_id: a.new_amount for a in adjustments if a.kind == "reduce_to"}

        flows = list(self.committed)
        start = self.request.request_date
        for s in self.series:
            if s.direction == "credit" and not self.config.project_income:
                continue
            anchor = s.last_event_id
            if anchor in stopped:
                continue
            base = s.forecast_amount(self.config.variable_policy)
            if anchor in reduced:
                base = reduced[anchor]
            for on in s.occurrences_between(start, self.horizon_end):
                amount = self._apply_expense_changes(s.category, base, on)
                signed = amount if s.direction == "credit" else -amount
                flows.append(
                    Flow(on, signed, f"{s.category} ({s.event_type})", anchor, s.category)
                )
        flows.sort(key=lambda f: (f.on, f.label))
        return flows

    def _apply_expense_changes(self, category: str, amount: float, on: date) -> float:
        """Apply any confirmed change to this category that is in force on ``on``."""
        for change in self.expense_changes:
            if change.get("category") != category:
                continue
            starts = change.get("effective_from")
            if starts and on < date.fromisoformat(starts):
                continue
            if change.get("new_amount") is not None:
                amount = float(change["new_amount"])
            elif change.get("increase_pct") is not None:
                amount = amount * (1 + float(change["increase_pct"]) / 100.0)
        return amount

    def trough(
        self,
        payments: tuple[tuple[date, float], ...] = (),
        adjustments: tuple[Adjustment, ...] = (),
    ) -> float:
        """Lowest projected balance over the horizon given extra payments."""
        flows = self.base_flows(adjustments)
        for on, amount in payments:
            flows.append(Flow(on, -amount, "request payment", source="plan"))
        flows.sort(key=lambda f: f.on)

        balance = self.start_balance
        low = balance
        for flow in flows:
            if flow.on > self.horizon_end:
                continue
            balance += flow.amount
            low = min(low, balance)
        return low

    def is_safe(
        self,
        payments: tuple[tuple[date, float], ...] = (),
        adjustments: tuple[Adjustment, ...] = (),
    ) -> bool:
        floor = self.profile.minimum_balance_to_keep
        return self.trough(payments, adjustments) >= floor - 1e-6

    # ---- the two headline numbers ---------------------------------------------

    def amount_safe_today(self) -> float:
        """Largest single payment on ``request_date`` that stays safe.

        No optional spending changes are applied, per the spec, and the result
        is capped at the requested amount.
        """
        headroom = self.trough() - self.profile.minimum_balance_to_keep
        return max(0.0, min(headroom, self.request.requested_amount))

    def earliest_full_payment_date(self) -> date | None:
        """First date in the horizon where the full amount is safe alone.

        This measures capacity only, so it ignores the user's method
        preferences and any optional spending change.
        """
        full = self.request.requested_amount
        day = self.request.request_date
        while day <= self.horizon_end:
            if self.is_safe(((day, full),)):
                return day
            day += timedelta(days=1)
        return None


def _flexibility_allows(series: recurrence.Series, profile: Profile) -> list[str]:
    """Which changes the user permits on this series."""
    if series.direction != "debit" or series.category in profile.protect:
        return []
    flex = series.flexibility or "fixed"
    kinds: list[str] = []
    if flex in ("stoppable", "reducible_or_stoppable") and series.category in profile.willing_to_stop:
        kinds.append("stop")
    if flex in ("reducible", "reducible_or_stoppable") and series.category in profile.willing_to_reduce:
        kinds.append("reduce_to")
    return kinds


def _income_flows(
    history: list[dict],
    scheduled_credits: list[Flow],
    rd: date,
    horizon_end: date,
    config: Config,
    income_update: dict | None = None,
) -> list[Flow]:
    """Project confirmed recurring income across the horizon.

    A scheduled credit is the strongest evidence available: it fixes both the
    amount and the pay day. When there is none we fall back to the settled
    history, which the dataset supplies on a monthly cadence.
    """
    flows: list[Flow] = list(scheduled_credits)
    credits = [
        (row["cash_date"], row["amount"], row["event_id"])
        for row in history
        if row["direction"] == "credit"
    ]
    streams = income.detect(credits)
    scheduled_dates = {f.on for f in scheduled_credits}
    claimed: set[date] = set()

    # (anchor_date, amount, anchor_id, period_days, day_of_month)
    projections: list[tuple[date, float, str, int | None, int | None]] = []
    for stream in streams:
        amount = stream.amount(config.income_policy)
        anchor_date, _, anchor_id = stream.last
        # A confirmed future credit on this stream's pay day supersedes the
        # projected amount and re-anchors the cadence.
        aligned = [
            f
            for f in scheduled_credits
            if stream.day_of_month is not None
            and abs(f.on.day - stream.day_of_month) <= income.DAY_TOLERANCE
        ]
        if aligned:
            confirmed = max(aligned, key=lambda f: f.on)
            anchor_date, amount, anchor_id = confirmed.on, confirmed.amount, confirmed.event_id or ""
            claimed.update(f.on for f in aligned)
        projections.append((anchor_date, amount, anchor_id, stream.period_days, stream.day_of_month))

    # A confirmed future credit that matches no detected stream still
    # establishes recurring income on its own: the user has too little settled
    # history for detection (a new job, a prorated first month) but the payment
    # itself is confirmed.
    for flow in scheduled_credits:
        if flow.on in claimed:
            continue
        projections.append((flow.on, flow.amount, flow.event_id or "", None, flow.on.day))

    if not config.extend_scheduled_income:
        return flows

    update = income_update or {}
    if update.get("income_stops"):
        # A contract ended with no renewal confirmed: only already-confirmed
        # credits survive, and nothing is projected beyond them.
        return flows

    confirmed_amount = update.get("new_recurring_amount")
    effective_from = (
        date.fromisoformat(update["effective_from"]) if update.get("effective_from") else None
    )
    one_off = update.get("one_off_next_amount")
    moved_to = (
        date.fromisoformat(update["next_payment_date"])
        if update.get("next_payment_date")
        else None
    )

    for anchor_date, amount, anchor_id, period_days, day_of_month in projections:
        cursor = anchor_date
        first = True
        while True:
            if day_of_month is not None:
                cursor = recurrence._add_month(cursor, day_of_month)
            else:
                cursor = cursor + timedelta(days=period_days)
            if cursor > horizon_end:
                break
            if cursor <= rd or cursor in scheduled_dates:
                continue

            on, value = cursor, amount
            if confirmed_amount is not None and (
                effective_from is None or cursor >= effective_from
            ):
                value = confirmed_amount
            if first:
                # A one-off adjustment or a moved pay date applies to the very
                # next payment only.
                if one_off is not None:
                    value = one_off
                if moved_to is not None and moved_to <= horizon_end:
                    on = moved_to
                first = False
            flows.append(Flow(on, value, "income", anchor_id, "salary", source="evidence"))
    return flows


def build(
    data: Dataset,
    request: Request,
    config: Config = Config(),
    patches: dict | None = None,
) -> Forecast:
    """Assemble the forecast for one request.

    ``patches`` carries evidence extracted from messages and images: filled-in
    amounts for blank events, cancellations, date shifts and confirmed income
    changes. It is applied here so that the LLM never touches the arithmetic.
    """
    profile = data.profiles[request.user_id]
    converter = Converter(data.exchange_rates)
    patch = patches or {}
    amount_overrides = {
        item["event_id"]: item["amount"] for item in patch.get("event_amounts", [])
    }
    amount_currencies = {
        item["event_id"]: item.get("currency") for item in patch.get("event_amounts", [])
    }
    cancelled: set[str] = set(patch.get("cancelled_event_ids", []))
    date_overrides = {
        item["event_id"]: date.fromisoformat(item["new_settlement_date"])
        for item in patch.get("event_date_changes", [])
    }
    income_update: dict = patch.get("income_update", {}) or {}
    expense_changes: list[dict] = patch.get("recurring_expense_changes", []) or []

    rows = data.events_by_user.get(request.user_id)
    descriptions: dict[str, str] = {}
    history: list[dict] = []
    committed: list[Flow] = []
    rd = request.request_date
    horizon_end = rd + timedelta(days=config.horizon_days)

    for row in (rows.to_dict("records") if rows is not None else []):
        eid = row["event_id"]
        descriptions[eid] = row["description"]
        if eid in cancelled or row["status"] in DEAD_STATUSES or row["direction"] == "non_cash":
            continue
        amount = amount_overrides.get(eid, row["amount"])
        if amount is None or (isinstance(amount, float) and amount != amount):
            # A blank amount with no supporting evidence must not become zero;
            # drop it from cash flow but keep it out of recurrence stats too.
            continue
        cash_date = date_overrides.get(eid) or row["cash_date"]
        if cash_date is None:
            continue
        currency = amount_currencies.get(eid) or row["currency"] if eid in amount_overrides else row["currency"]
        amount = converter.convert(float(amount), currency, profile.home_currency, cash_date)

        if cash_date <= rd and row["status"] == "settled":
            history.append(
                {
                    "event_id": eid,
                    "event_type": row["event_type"],
                    "category": row["category"],
                    "direction": row["direction"],
                    "cash_date": cash_date,
                    "amount": amount,
                    "flexibility": row["flexibility"],
                    "minimum_allowed_amount": row["minimum_allowed_amount"],
                }
            )
            continue

        if cash_date <= rd:
            continue  # a stale pending row that should already be reflected
        if row["status"] == "pending":
            if row["direction"] == "debit" and config.reserve_pending_debits:
                committed.append(
                    Flow(cash_date, -amount, f"pending {row['category']}", eid, row["category"],
                         source="committed")
                )
            continue  # pending credits never count until they settle
        if row["status"] == "scheduled":
            signed = amount if row["direction"] == "credit" else -amount
            if row["direction"] == "credit" and not config.count_scheduled_credits:
                continue
            committed.append(
                Flow(cash_date, signed, f"scheduled {row['category']}", eid, row["category"],
                     source="committed")
            )

    # Income is handled separately from ordinary recurrence detection. A user
    # may have only one or two settled salary rows (a prorated first month, a
    # recent job change) yet a confirmed scheduled payment that establishes both
    # the amount and the pay day going forward.
    series = [s for s in recurrence.detect(history) if s.direction == "debit"]
    scheduled_credits = [f for f in committed if f.amount > 0]
    committed = [f for f in committed if f.amount <= 0]
    if config.project_income:
        committed.extend(
            _income_flows(history, scheduled_credits, rd, horizon_end, config, income_update)
        )
    else:
        committed.extend(scheduled_credits)

    adjustable: list[Adjustment] = []
    for s in series:
        for kind in _flexibility_allows(s, profile):
            projected = s.forecast_amount(config.variable_policy)
            if kind == "stop":
                adjustable.append(
                    Adjustment(s.last_event_id, "stop", None, s.category, projected)
                )
            else:
                floor = s.minimum_allowed_amount
                if floor is None or floor >= projected:
                    continue
                adjustable.append(
                    Adjustment(s.last_event_id, "reduce_to", floor, s.category, projected - floor)
                )

    return Forecast(
        profile=profile,
        request=request,
        series=series,
        committed=committed,
        config=config,
        start_balance=profile.current_available_balance,
        horizon_end=horizon_end,
        adjustable=adjustable,
        expense_changes=expense_changes,
        descriptions=descriptions,
    )
