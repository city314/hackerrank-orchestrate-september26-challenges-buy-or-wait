"""Currency conversion against the fixed, dated rate table."""

from __future__ import annotations

from bisect import bisect_right
from datetime import date


class Converter:
    """Converts foreign-currency amounts into a user's home currency.

    Rates are dated. For a cash event we use the rate in force on its
    settlement date: the latest row on or before that date, falling back to the
    earliest row when the event predates the table.
    """

    def __init__(self, rates: dict[tuple[str, str], list[tuple[date, float]]]):
        self.rates = rates

    def _direct(self, src: str, dst: str, on: date) -> float | None:
        series = self.rates.get((src, dst))
        if not series:
            return None
        idx = bisect_right([d for d, _ in series], on)
        return series[idx - 1][1] if idx else series[0][1]

    def rate(self, src: str, dst: str, on: date) -> float:
        if src == dst:
            return 1.0
        direct = self._direct(src, dst, on)
        if direct is not None:
            return direct
        # The table stores one direction for some pairs; invert when needed.
        inverse = self._direct(dst, src, on)
        if inverse:
            return 1.0 / inverse
        # Last resort: hop through a currency quoted against both.
        for mid in {c for pair in self.rates for c in pair}:
            a, b = self._direct(src, mid, on), self._direct(mid, dst, on)
            if a is not None and b is not None:
                return a * b
        raise KeyError(f"no exchange rate for {src}->{dst} on {on}")

    def convert(self, amount: float, src: str, dst: str, on: date) -> float:
        return amount if src == dst else amount * self.rate(src, dst, on)
