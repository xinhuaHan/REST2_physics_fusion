from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SOLAR_CONSTANT = 1361.0


@dataclass(frozen=True)
class SiteConfig:
    latitude: float
    longitude: float
    altitude_m: float = 0.0
    timezone: str = "Asia/Shanghai"


def standard_pressure_pa(altitude_m: float) -> float:
    return float(101325.0 * np.power(1.0 - 2.25577e-5 * altitude_m, 5.25588))


def localize_time(values: pd.Series | pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    times = pd.to_datetime(values)
    if isinstance(times, pd.Series):
        if times.dt.tz is None:
            return pd.DatetimeIndex(times).tz_localize(timezone)
        return pd.DatetimeIndex(times).tz_convert(timezone)
    if times.tz is None:
        return pd.DatetimeIndex(times).tz_localize(timezone)
    return pd.DatetimeIndex(times).tz_convert(timezone)


def add_solar_geometry(df: pd.DataFrame, timestamp_col: str, site: SiteConfig) -> pd.DataFrame:
    out = df.copy()
    times = localize_time(out[timestamp_col], site.timezone)
    day_of_year = times.dayofyear.to_numpy(dtype=float)
    hour = (
        times.hour.to_numpy(dtype=float)
        + times.minute.to_numpy(dtype=float) / 60.0
        + times.second.to_numpy(dtype=float) / 3600.0
    )

    gamma = 2.0 * np.pi * (day_of_year - 1.0) / 365.0
    equation_of_time_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    utc_offset_hours = times[0].utcoffset().total_seconds() / 3600.0
    local_standard_meridian = 15.0 * utc_offset_hours
    time_offset_min = equation_of_time_min + 4.0 * (site.longitude - local_standard_meridian)
    solar_time_hour = hour + time_offset_min / 60.0
    hour_angle = np.deg2rad(15.0 * (solar_time_hour - 12.0))
    lat = np.deg2rad(site.latitude)

    cos_zenith = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    mu0 = np.clip(cos_zenith, 0.0, 1.0)
    apparent_zenith = np.rad2deg(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    apparent_elevation = 90.0 - apparent_zenith
    eccentricity = (
        1.00011
        + 0.034221 * np.cos(gamma)
        + 0.00128 * np.sin(gamma)
        + 0.000719 * np.cos(2.0 * gamma)
        + 0.000077 * np.sin(2.0 * gamma)
    )

    out["timestamp_local"] = times.astype(str)
    out["day_of_year"] = day_of_year.astype(int)
    out["apparent_zenith"] = apparent_zenith
    out["apparent_elevation"] = apparent_elevation
    out["mu0_target"] = mu0
    out["dni_extra"] = SOLAR_CONSTANT * eccentricity
    return out
