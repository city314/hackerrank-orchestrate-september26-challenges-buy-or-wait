"""Dataset loading and typed access.

Everything downstream reads the dataset through this module so that paths,
dtypes and date parsing are decided in exactly one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# Statuses that describe money that has already moved.
SETTLED = "settled"
# Money that is committed but has not moved yet.
FUTURE_STATUSES = ("pending", "scheduled")
# Records that must never affect the forecast.
DEAD_STATUSES = ("cancelled", "failed", "unrealized")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").dt.date


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _split_list(value) -> list[str]:
    """Split the pipe-delimited profile columns into a clean list."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: list[str] = field(default_factory=list)
    protect: list[str] = field(default_factory=list)
    willing_to_reduce: list[str] = field(default_factory=list)
    willing_to_stop: list[str] = field(default_factory=list)
    payment_methods: list[str] = field(default_factory=list)
    max_installment_months: int | None = None

    def accepts(self, method: str) -> bool:
        return method in self.payment_methods


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: float
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: float
    total_payable_amount: float

    @property
    def sort_key(self) -> int:
        """Numeric suffix of the option id, used as the documented tie-breaker."""
        return int(self.payment_option_id.rsplit("_", 1)[-1])


class Dataset:
    """Lazily-parsed view over the participant-facing CSV files."""

    def __init__(self, dataset_dir: str = DATASET_DIR):
        self.dir = dataset_dir

    def _read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dir, name), dtype=str, keep_default_na=False)

    @cached_property
    def events(self) -> pd.DataFrame:
        df = self._read("financial_events.csv")
        df["amount"] = pd.to_numeric(df["amount"].replace("", None), errors="coerce")
        df["minimum_allowed_amount"] = pd.to_numeric(
            df["minimum_allowed_amount"].replace("", None), errors="coerce"
        )
        df["event_date"] = _to_date(df["event_date"].replace("", None))
        df["settlement_date"] = _to_date(df["settlement_date"].replace("", None))
        # A missing settlement date means the event never settled on its own;
        # fall back to the event date so ordering still works.
        df["cash_date"] = df["settlement_date"].fillna(df["event_date"])
        for col in ("linked_event_id", "flexibility"):
            df[col] = df[col].replace("", None)
        return df

    @cached_property
    def profiles(self) -> dict[str, Profile]:
        df = self._read("financial_profiles.csv")
        out: dict[str, Profile] = {}
        for row in df.to_dict("records"):
            months = row["max_installment_months"].strip()
            out[row["user_id"]] = Profile(
                user_id=row["user_id"],
                home_currency=row["home_currency"],
                current_available_balance=float(row["current_available_balance"]),
                minimum_balance_to_keep=float(row["minimum_balance_to_keep"]),
                financial_priorities=_split_list(row["financial_priorities"]),
                protect=_split_list(row["expense_categories_to_protect"]),
                willing_to_reduce=_split_list(row["expense_categories_user_is_willing_to_reduce"]),
                willing_to_stop=_split_list(row["expense_categories_user_is_willing_to_stop"]),
                payment_methods=_split_list(row["payment_methods_user_will_consider"]),
                max_installment_months=int(float(months)) if months else None,
            )
        return out

    def _requests_from(self, name: str) -> list[Request]:
        df = self._read(name)
        return [
            Request(
                request_id=r["request_id"],
                user_id=r["user_id"],
                request_date=date.fromisoformat(r["request_date"]),
                request_type=r["request_type"],
                requested_amount=float(r["requested_amount"]),
                desired_completion_date=date.fromisoformat(r["desired_completion_date"]),
                allows_partial_payment=str(r["allows_partial_payment"]).strip().lower()
                in ("true", "1", "yes"),
                request_text=r["request_text"],
            )
            for r in df.to_dict("records")
        ]

    @cached_property
    def requests(self) -> list[Request]:
        return self._requests_from("requests.csv")

    @cached_property
    def sample_requests(self) -> list[Request]:
        return self._requests_from("sample_requests.csv")

    @cached_property
    def sample_answers(self) -> pd.DataFrame:
        """The 25 solved examples, indexed by request id."""
        df = self._read("sample_requests.csv")
        df["amount_safe_to_pay"] = pd.to_numeric(df["amount_safe_to_pay"])
        return df.set_index("request_id")

    @cached_property
    def payment_options(self) -> dict[str, list[PaymentOption]]:
        df = self._read("request_payment_options.csv")
        out: dict[str, list[PaymentOption]] = {}
        for r in df.to_dict("records"):
            freq = r["payment_frequency_days"].strip()
            opt = PaymentOption(
                payment_option_id=r["payment_option_id"],
                request_id=r["request_id"],
                payment_method=r["payment_method"],
                payment_amount=float(r["payment_amount"]),
                number_of_payments=int(float(r["number_of_payments"])),
                first_payment_date=date.fromisoformat(r["first_payment_date"]),
                payment_frequency_days=int(float(freq)) if freq else None,
                financing_fee=float(r["financing_fee"] or 0),
                total_payable_amount=float(r["total_payable_amount"]),
            )
            out.setdefault(opt.request_id, []).append(opt)
        for options in out.values():
            options.sort(key=lambda o: o.sort_key)
        return out

    @cached_property
    def messages(self) -> pd.DataFrame:
        df = self._read("messages.csv")
        df["sent_at_ts"] = pd.to_datetime(df["sent_at"], errors="coerce", utc=True)
        for col in ("request_id", "related_event_id"):
            df[col] = df[col].replace("", None)
        return df

    @cached_property
    def images(self) -> pd.DataFrame:
        df = self._read("images.csv")
        for col in ("request_id", "related_event_id"):
            df[col] = df[col].replace("", None)
        df["path"] = df["image_id"].map(
            lambda i: os.path.join(self.dir, "media", "images", f"{i}.png")
        )
        return df

    @cached_property
    def exchange_rates(self) -> dict[tuple[str, str], list[tuple[date, float]]]:
        """Rates keyed by (from, to), each a date-sorted list."""
        df = self._read("exchange_rates.csv")
        out: dict[tuple[str, str], list[tuple[date, float]]] = {}
        for r in df.to_dict("records"):
            key = (r["from_currency"], r["to_currency"])
            out.setdefault(key, []).append((date.fromisoformat(r["rate_date"]), float(r["rate"])))
        for series in out.values():
            series.sort()
        return out

    @cached_property
    def events_by_user(self) -> dict[str, pd.DataFrame]:
        return {uid: g.copy() for uid, g in self.events.groupby("user_id")}
