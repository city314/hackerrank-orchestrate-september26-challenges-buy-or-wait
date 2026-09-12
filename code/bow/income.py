"""Identify which income a user can actually count on.

The dataset puts several distinct income streams under the same ``salary``
category: a contractual payroll credit on a stable pay day, and — for some
users — a commission or bonus that lands on a different day with a different
amount every month. The spec is explicit that bonuses and commissions must not
be counted until they settle, so the two must be separated before anything is
projected forward.

The separating signal is structural rather than lexical: contractual pay
repeats on a fixed cadence for a constant amount, while commission does not.
Gig-economy users have no constant stream at all, and for them the variable
payouts *are* the income, so they are projected conservatively instead of
being dropped.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date

# Cadence bounds shared with the expense detector.
MONTHLY_MIN, MONTHLY_MAX = 26, 33
MIN_OCCURRENCES = 3
# Day-of-month jitter still considered the same pay day.
DAY_TOLERANCE = 3


@dataclass
class Stream:
    """One coherent income stream."""

    events: list[tuple[date, float, str]]  # (date, amount, event_id)
    period_days: int | None
    day_of_month: int | None

    @property
    def amounts(self) -> list[float]:
        return [a for _, a, _ in self.events]

    @property
    def is_constant(self) -> bool:
        """A contractual amount repeats to the cent."""
        return len(set(round(a, 2) for a in self.amounts)) == 1

    @property
    def last(self) -> tuple[date, float, str]:
        return self.events[-1]

    def amount(self, policy: str) -> float:
        amounts = self.amounts
        if self.is_constant:
            return amounts[-1]
        if policy == "min":
            return min(amounts)
        if policy == "mean":
            return statistics.fmean(amounts)
        if policy == "median":
            return statistics.median(amounts)
        if policy == "last":
            return amounts[-1]
        if policy.startswith("min"):
            return min(amounts[-int(policy[3:]) :])
        if policy.startswith("mean"):
            return statistics.fmean(amounts[-int(policy[4:]) :])
        return statistics.fmean(amounts)


def _monthly_clusters(events: list[tuple[date, float, str]]) -> tuple[list[Stream], list]:
    """Split events into monthly streams keyed by pay day, plus leftovers."""
    buckets: dict[int, list] = {}
    for entry in events:
        day = entry[0].day
        for key in buckets:
            if abs(key - day) <= DAY_TOLERANCE:
                buckets[key].append(entry)
                break
        else:
            buckets[day] = [entry]

    streams, leftover = [], []
    for day, entries in buckets.items():
        entries.sort()
        gaps = [(b[0] - a[0]).days for a, b in zip(entries, entries[1:])]
        monthly = (
            len(entries) >= MIN_OCCURRENCES
            and gaps
            and all(MONTHLY_MIN <= g <= MONTHLY_MAX for g in gaps)
        )
        if monthly:
            streams.append(Stream(entries, None, statistics.mode([e[0].day for e in entries])))
        else:
            leftover.extend(entries)
    return streams, leftover


def _periodic_stream(events: list[tuple[date, float, str]]) -> Stream | None:
    """Treat the whole set as one fixed-cadence stream, if it is regular."""
    events = sorted(events)
    if len(events) < MIN_OCCURRENCES:
        return None
    gaps = [(b[0] - a[0]).days for a, b in zip(events, events[1:])]
    if not gaps or min(gaps) <= 0:
        return None
    median = statistics.median(gaps)
    if not all(abs(g - median) <= max(1.0, 0.35 * median) for g in gaps):
        return None
    if MONTHLY_MIN <= median <= MONTHLY_MAX:
        return Stream(events, None, statistics.mode([e[0].day for e in events]))
    return Stream(events, int(round(median)), None)


def detect(credits: list[tuple[date, float, str]]) -> list[Stream]:
    """Return the income streams a forecast may rely on.

    Constant-amount streams win outright: when a user has contractual pay, the
    variable streams alongside it are commission or bonus and the spec says not
    to count them.
    """
    if not credits:
        return []
    monthly, leftover = _monthly_clusters(sorted(credits))
    streams = list(monthly)
    if leftover:
        extra = _periodic_stream(leftover)
        if extra is not None:
            streams.append(extra)

    constant = [s for s in streams if s.is_constant]
    return constant if constant else streams
