"""Detect recurring cash-flow series from a user's settled history.

The dataset gives roughly six months of history per user. Bills repeat on a
day-of-month cadence; groceries, transport and dining repeat on a fixed
day-count cadence that differs per user. One-off noise (a reversed card
charge, a cancelled authorisation) must not be mistaken for a series, so a
group only becomes a series when its spacing is genuinely regular.
"""

from __future__ import annotations

import statistics
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

# A series needs at least this many past occurrences before we trust it.
MIN_OCCURRENCES = 3
# Gaps this close to a month are treated as a day-of-month cadence.
MONTHLY_MIN, MONTHLY_MAX = 26, 33
# A day-count cadence is accepted when every gap is within this factor of the
# median gap. Real weekly/biweekly series in this dataset are exact, so the
# tolerance only needs to absorb the occasional shifted date.
GAP_TOLERANCE = 0.35


@dataclass
class Occurrence:
    """One past instance of a recurring series."""

    event_id: str
    on: date
    amount: float


@dataclass
class Series:
    """A detected recurring inflow or outflow."""

    key: tuple[str, str]
    event_type: str
    category: str
    direction: str  # "debit" or "credit"
    period_days: int | None  # None for a day-of-month cadence
    day_of_month: int | None
    occurrences: list[Occurrence]
    flexibility: str
    minimum_allowed_amount: float | None

    @property
    def last_date(self) -> date:
        return self.occurrences[-1].on

    @property
    def amounts(self) -> list[float]:
        return [o.amount for o in self.occurrences]

    @property
    def is_variable(self) -> bool:
        """True when the series amount moves between occurrences."""
        return len(set(round(a, 2) for a in self.amounts)) > 1

    @property
    def last_event_id(self) -> str:
        return self.occurrences[-1].event_id

    def forecast_amount(self, policy: str) -> float:
        """Project the next amount for this series under a forecast policy.

        Fixed-amount series ignore the policy entirely — there is nothing to
        estimate. Variable series are where the policy matters, and the problem
        statement asks for a conservative estimate of essential variable
        spending.
        """
        amounts = self.amounts
        if not self.is_variable:
            return amounts[-1]
        if policy == "last":
            return amounts[-1]
        if policy == "max":
            return max(amounts)
        if policy == "mean":
            return statistics.fmean(amounts)
        if policy == "median":
            return statistics.median(amounts)
        if policy == "trimmed":
            # Drop the extreme observations, then average what is left.
            ordered = sorted(amounts)
            trimmed = ordered[1:-1] if len(ordered) >= 4 else ordered
            return statistics.fmean(trimmed)
        if policy.startswith("max"):
            return max(amounts[-int(policy[3:]) :])
        if policy.startswith("min"):
            return min(amounts[-int(policy[3:]) :])
        if policy.startswith("mean"):
            return statistics.fmean(amounts[-int(policy[4:]) :])
        raise ValueError(f"unknown forecast policy: {policy}")

    def occurrences_between(self, start: date, end: date, include_start: bool = False) -> list[date]:
        """Project future occurrence dates in the window up to and including ``end``.

        An occurrence landing exactly on ``start`` has not settled yet — the
        series' latest recorded occurrence is strictly earlier — so it is real
        upcoming spending, not something already inside the opening balance.
        """
        out: list[date] = []
        if self.day_of_month is not None:
            cursor = self.last_date
            while True:
                cursor = _add_month(cursor, self.day_of_month)
                if cursor > end:
                    break
                if cursor > start or (include_start and cursor == start):
                    out.append(cursor)
        else:
            cursor = self.last_date
            while True:
                cursor = cursor + timedelta(days=self.period_days)
                if cursor > end:
                    break
                if cursor > start or (include_start and cursor == start):
                    out.append(cursor)
        return out


def _add_month(d: date, day_of_month: int) -> date:
    """Advance one calendar month, clamping to the target day of month."""
    year, month = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(year, month, min(day_of_month, monthrange(year, month)[1]))


def _cadence(dates: list[date]) -> tuple[int | None, int | None]:
    """Infer a cadence from occurrence dates.

    Returns ``(period_days, day_of_month)`` with exactly one of the two set,
    or ``(None, None)`` when the spacing is too irregular to trust.
    """
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    if not gaps or min(gaps) <= 0:
        return None, None
    median = statistics.median(gaps)

    if MONTHLY_MIN <= median <= MONTHLY_MAX:
        if all(MONTHLY_MIN <= g <= MONTHLY_MAX for g in gaps):
            # Most bills land on a stable day of month; a few drift by a day.
            days = [d.day for d in dates]
            return None, statistics.mode(days)
        return None, None

    if all(abs(g - median) <= max(1.0, GAP_TOLERANCE * median) for g in gaps):
        return int(round(median)), None
    return None, None


def _drop_outliers(rows: list[dict], factor: float = 3.0) -> list[dict]:
    """Remove amounts far outside the group's usual range.

    A single bulk purchase filed under ``groceries`` is a one-off, not a larger
    instalment of the weekly shop, and letting it into the series would inflate
    every projected occurrence.
    """
    amounts = [r["amount"] for r in rows if r["amount"]]
    if len(amounts) < MIN_OCCURRENCES:
        return rows
    middle = statistics.median(amounts)
    if middle <= 0:
        return rows
    kept = [r for r in rows if middle / factor <= r["amount"] <= middle * factor]
    return kept if len(kept) >= MIN_OCCURRENCES else rows


def _on_grid(rows: list[dict]) -> list[dict]:
    """Keep only the occurrences that sit on the group's dominant cadence."""
    dates = [r["cash_date"] for r in rows]
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    if not gaps:
        return rows
    step = statistics.mode(gaps)
    if step <= 0:
        return rows
    anchor = dates[-1]
    return [r for r in rows if (anchor - r["cash_date"]).days % step == 0]


def detect(history: list[dict]) -> list[Series]:
    """Group settled history into recurring series.

    ``history`` rows must already be in home currency, filtered to events that
    actually moved money, and restricted to dates before the request date.
    Grouping is by (event_type, category): within a user the dataset uses one
    cadence per category, while descriptions vary between occurrences.
    """
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in history:
        groups.setdefault((row["event_type"], row["category"], row["direction"]), []).append(row)

    series: list[Series] = []
    for (event_type, category, direction), rows in groups.items():
        rows.sort(key=lambda r: r["cash_date"])
        rows = _drop_outliers(rows)
        # Two rows on the same day are two separate purchases, not a larger
        # instalment of the series: keep the first and treat the rest as one-offs.
        deduped: list[dict] = []
        for row in rows:
            if not deduped or deduped[-1]["cash_date"] != row["cash_date"]:
                deduped.append(dict(row))
        if len(deduped) < MIN_OCCURRENCES:
            continue

        period, day_of_month = _cadence([r["cash_date"] for r in deduped])
        if period is None and day_of_month is None:
            # A one-off sitting next to a real series can hide its cadence.
            # Retry on just the occurrences that sit on a regular grid.
            deduped = _on_grid(deduped)
            if len(deduped) < MIN_OCCURRENCES:
                continue
            period, day_of_month = _cadence([r["cash_date"] for r in deduped])
            if period is None and day_of_month is None:
                continue

        last = deduped[-1]
        series.append(
            Series(
                key=(event_type, category),
                event_type=event_type,
                category=category,
                direction=direction,
                period_days=period,
                day_of_month=day_of_month,
                occurrences=[
                    Occurrence(r["event_id"], r["cash_date"], r["amount"]) for r in deduped
                ],
                flexibility=last.get("flexibility") or "fixed",
                minimum_allowed_amount=last.get("minimum_allowed_amount"),
            )
        )
    series.sort(key=lambda s: s.key)
    return series
