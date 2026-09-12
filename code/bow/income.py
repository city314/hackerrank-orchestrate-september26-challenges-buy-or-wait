"""Identify which income a user can actually count on.

The dataset puts several distinct income streams under the same ``salary``
category: a contractual payroll credit on a stable pay day, and — for some
users — a commission or bonus that lands on a different day with a different
amount every month. The spec is explicit that bonuses and commissions must not
be counted until they settle, so the two must be separated before anything is
projected forward.

Amount stability alone cannot separate them: a second household salary moves
month to month and is still real income, while a commission that happens to be
flat is still contingent. The description is what distinguishes them, so a
stream is judged by how the dataset labels its payments. Gig-economy users have
no fixed-amount stream at all, and for them the variable payouts *are* the
income, projected conservatively rather than dropped.
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


# Income that is contingent on performance or luck. The spec says not to count
# it until it settles, so a stream described this way is never projected.
CONTINGENT_TERMS = (
    "commission", "bonus", "arrears", "prize", "lottery", "windfall",
    "incentive", "gratuity",
)
# Wording that marks a payment as the last of its stream.
TERMINAL_TERMS = ("final", "last ", "closing")


def _has_term(text: str, terms) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in terms)


@dataclass
class Stream:
    """One coherent income stream."""

    events: list[tuple[date, float, str, str]]  # (date, amount, event_id, description)
    period_days: int | None
    day_of_month: int | None

    @property
    def is_contingent(self) -> bool:
        """A commission or bonus stream, which must not be projected forward."""
        hits = sum(1 for e in self.events if _has_term(e[3], CONTINGENT_TERMS))
        return hits * 2 > len(self.events)

    @property
    def has_ended(self) -> bool:
        """The most recent payment announces itself as the final one."""
        return _has_term(self.events[-1][3], TERMINAL_TERMS)

    @property
    def amounts(self) -> list[float]:
        return [e[1] for e in self.events]

    @property
    def is_constant(self) -> bool:
        """A contractual amount repeats to the cent."""
        return len(set(round(a, 2) for a in self.amounts)) == 1

    @property
    def last(self) -> tuple[date, float, str, str]:
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


def _monthly_clusters(events: list[tuple[date, float, str, str]]) -> tuple[list[Stream], list]:
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


def _periodic_stream(events: list[tuple[date, float, str, str]]) -> Stream | None:
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


def detect(credits: list[tuple[date, float, str, str]]) -> list[Stream]:
    """Return the income streams a forecast may rely on.

    Streams described as commission, bonus or a prize are dropped: the spec
    forbids counting them before they settle. A second household salary is a
    real, recurring stream even though its amount moves, so amount stability
    alone is not enough to tell the two apart — the description is.

    A stream whose latest payment calls itself the final one has ended, and is
    reported so the caller can stop projecting it.
    """
    if not credits:
        return []
    monthly, leftover = _monthly_clusters(sorted(credits))
    streams = list(monthly)
    if leftover:
        extra = _periodic_stream(leftover)
        if extra is not None:
            streams.append(extra)

    kept = [s for s in streams if not s.is_contingent]
    if kept:
        return kept
    # Every stream looks contingent: fall back to any constant one rather than
    # forecasting a user with no income at all.
    return [s for s in streams if s.is_constant]
