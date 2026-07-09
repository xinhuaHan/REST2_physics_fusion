from __future__ import annotations

import pandas as pd


def to_naive_local_datetime(values: pd.Series) -> pd.Series:
    """Parse timestamps and drop timezone labels while preserving local clock time."""

    index = values.index
    try:
        parsed = pd.to_datetime(values, errors="coerce")
    except ValueError:
        parsed = pd.Series([_parse_one_timestamp(value) for value in values], index=index)

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_localize(None)

    parsed = pd.Series(parsed, index=index)
    if parsed.dtype == "object":
        parsed = pd.Series([_drop_timezone(value) for value in parsed], index=index)
    return pd.to_datetime(parsed, errors="coerce")


def _parse_one_timestamp(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    return _drop_timezone(pd.Timestamp(value))


def _drop_timezone(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp
