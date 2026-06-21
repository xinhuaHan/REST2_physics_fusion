from __future__ import annotations

import pandas as pd


WEATHER_COLUMNS = ["TEMP", "WS", "WD", "PREC", "PWAT", "SDWE"]


def merge_station_with_weather(
    station: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    timestamp_column: str = "dtime",
    tolerance_minutes: int = 60,
    weather_label: str = "weather_history",
) -> pd.DataFrame:
    """Attach nearest weather rows to station observations.

    Station files keep their strict raw schema (`dtime, observe_power,
    observe_ghi`). Weather fields are explicitly joined and marked, so
    downstream physics features know these are not native station columns.
    """

    station_work = station.copy()
    weather_work = weather.copy()
    station_work[timestamp_column] = pd.to_datetime(station_work[timestamp_column])
    weather_work[timestamp_column] = pd.to_datetime(weather_work[timestamp_column])
    station_work = station_work.sort_values(timestamp_column)
    weather_work = weather_work.sort_values(timestamp_column)

    keep_weather = [timestamp_column, *[col for col in WEATHER_COLUMNS if col in weather_work.columns]]
    merged = pd.merge_asof(
        station_work,
        weather_work[keep_weather],
        on=timestamp_column,
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    has_weather = merged[[col for col in WEATHER_COLUMNS if col in merged.columns]].notna().any(axis=1)
    merged["weather_is_joined"] = has_weather.astype(float)
    merged["weather_join_status"] = has_weather.map(
        {True: f"joined_{weather_label}", False: "missing_after_join"}
    )
    return merged
