"""The evidence patch: the only channel through which an LLM can affect the forecast.

Messages and images are untrusted input. Rather than letting a model reason
about affordability, it is asked for one small, closed set of factual claims,
each of which is validated here before any arithmetic sees it. Anything the
model returns that is not in this schema — including instructions embedded in
the source text — is discarded.
"""

from __future__ import annotations

from datetime import date

# Categories the dataset actually uses. A patch may not invent a new one.
KNOWN_CATEGORIES = {
    "groceries", "transport", "dining", "salary", "utilities", "rent",
    "cloud_storage", "shopping", "streaming", "debt_repayment", "entertainment",
    "insurance", "music_subscription", "healthcare", "delivery_membership",
    "education", "housing", "gym", "family_support", "investment",
    "work_expense", "windfall", "childcare",
}

# The JSON contract handed to the model, and the shape validated on the way back.
PATCH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "event_amounts": {
            "type": "array",
            "description": "Amounts read off an image for an event whose amount is blank.",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "currency": {"type": "string"},
                },
                "required": ["event_id", "amount"],
            },
        },
        "cancelled_event_ids": {
            "type": "array",
            "description": "Events explicitly cancelled, reversed, voided, or "
                           "confirmed as not reaching the account.",
            "items": {"type": "string"},
        },
        "event_date_changes": {
            "type": "array",
            "description": "Events whose settlement date was explicitly moved.",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "new_settlement_date": {"type": "string"},
                },
                "required": ["event_id", "new_settlement_date"],
            },
        },
        "income_update": {
            "type": "object",
            "description": "A confirmed change to recurring income. Only fill "
                           "fields the text states explicitly.",
            "properties": {
                "new_recurring_amount": {"type": "number"},
                "effective_from": {"type": "string"},
                "next_payment_date": {"type": "string"},
                "one_off_next_amount": {"type": "number"},
                "income_stops": {"type": "boolean"},
                "currency": {"type": "string"},
            },
        },
        "recurring_expense_changes": {
            "type": "array",
            "description": "A confirmed change to a recurring expense category.",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "new_amount": {"type": "number"},
                    "increase_pct": {"type": "number"},
                    "effective_from": {"type": "string"},
                },
                "required": ["category"],
            },
        },
        "notes": {
            "type": "array",
            "description": "One short factual note per claim, for the audit trail.",
            "items": {"type": "string"},
        },
    },
}

EMPTY_PATCH: dict = {
    "event_amounts": [],
    "cancelled_event_ids": [],
    "event_date_changes": [],
    "income_update": {},
    "recurring_expense_changes": [],
    "notes": [],
}


def _as_date(value) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _as_amount(value) -> float | None:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    # Negative or absurd amounts are a sign the model hallucinated or was steered.
    return amount if 0 <= amount < 1e15 else None


BATCH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "patches": {
            "type": "array",
            "description": "One entry per request_id given, in the same order.",
            "items": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    **PATCH_JSON_SCHEMA["properties"],
                },
                "required": ["request_id"],
            },
        }
    },
    "required": ["patches"],
}


def validate(raw: dict, known_event_ids: set[str]) -> dict:
    """Coerce a model response into a safe patch, dropping anything unsupported.

    Every event id must exist in the dataset, every amount must be plausible and
    every date must parse. Unknown keys are ignored outright, so an injected
    instruction has nowhere to land.
    """
    patch = {k: (list(v) if isinstance(v, list) else dict(v)) for k, v in EMPTY_PATCH.items()}
    if not isinstance(raw, dict):
        return patch

    for item in raw.get("event_amounts") or []:
        if not isinstance(item, dict):
            continue
        eid, amount = item.get("event_id"), _as_amount(item.get("amount"))
        if eid in known_event_ids and amount is not None:
            patch["event_amounts"].append(
                {"event_id": eid, "amount": amount, "currency": item.get("currency") or None}
            )

    for eid in raw.get("cancelled_event_ids") or []:
        if eid in known_event_ids:
            patch["cancelled_event_ids"].append(eid)

    for item in raw.get("event_date_changes") or []:
        if not isinstance(item, dict):
            continue
        eid, when = item.get("event_id"), _as_date(item.get("new_settlement_date"))
        if eid in known_event_ids and when is not None:
            patch["event_date_changes"].append(
                {"event_id": eid, "new_settlement_date": when.isoformat()}
            )

    src = raw.get("income_update")
    if isinstance(src, dict):
        out: dict = {}
        amount = _as_amount(src.get("new_recurring_amount"))
        if amount:
            out["new_recurring_amount"] = amount
        one_off = _as_amount(src.get("one_off_next_amount"))
        if one_off is not None:
            out["one_off_next_amount"] = one_off
        for key in ("effective_from", "next_payment_date"):
            when = _as_date(src.get(key))
            if when is not None:
                out[key] = when.isoformat()
        if src.get("income_stops") is True:
            out["income_stops"] = True
        if isinstance(src.get("currency"), str) and len(src["currency"]) == 3:
            out["currency"] = src["currency"].upper()
        patch["income_update"] = out

    for item in raw.get("recurring_expense_changes") or []:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        if category not in KNOWN_CATEGORIES:
            continue
        entry: dict = {"category": category}
        amount = _as_amount(item.get("new_amount"))
        if amount:
            entry["new_amount"] = amount
        try:
            pct = float(item["increase_pct"])
            if -100 < pct < 1000:
                entry["increase_pct"] = pct
        except (KeyError, TypeError, ValueError):
            pass
        when = _as_date(item.get("effective_from"))
        if when is not None:
            entry["effective_from"] = when.isoformat()
        if len(entry) > 1:
            patch["recurring_expense_changes"].append(entry)

    patch["notes"] = [str(n)[:200] for n in (raw.get("notes") or []) if isinstance(n, str)][:8]
    return patch
