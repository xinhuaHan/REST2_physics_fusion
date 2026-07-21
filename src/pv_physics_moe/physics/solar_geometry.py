from __future__ import annotations
import numpy as np
import pandas as pd


def solar_geometry(timestamps: pd.Series, latitude: float, longitude: float, timezone: str) -> pd.DataFrame:
    """NOAA-style solar geometry without a pvlib runtime dependency."""
    times = pd.to_datetime(timestamps, errors="coerce")
    if times.isna().any():
        raise ValueError("timestamps contain unparseable values")
    index = pd.DatetimeIndex(times)
    if index.tz is None:
        index = index.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
    else:
        index = index.tz_convert(timezone)
    day = index.dayofyear.to_numpy(dtype=float)
    hour = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0 + index.second.to_numpy(dtype=float) / 3600.0
    gamma = 2.0 * np.pi / 365.0 * (day - 1.0 + (hour - 12.0) / 24.0)
    equation = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    declination = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma) - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma) - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    utc_offsets = np.asarray([item.utcoffset().total_seconds() / 3600.0 for item in index.to_pydatetime()])
    time_offset = equation + 4.0 * float(longitude) - 60.0 * utc_offsets
    true_solar_minutes = (hour * 60.0 + time_offset) % 1440.0
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    latitude_rad = np.deg2rad(float(latitude))
    mu0 = np.sin(latitude_rad) * np.sin(declination) + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    mu0 = np.clip(mu0, -1.0, 1.0)
    zenith = np.rad2deg(np.arccos(mu0))
    dni_extra = 1361.0 * (1.0 + 0.033 * np.cos(2.0 * np.pi * day / 365.0))
    return pd.DataFrame({"mu0": mu0, "apparent_zenith": zenith, "dni_extra": dni_extra})

