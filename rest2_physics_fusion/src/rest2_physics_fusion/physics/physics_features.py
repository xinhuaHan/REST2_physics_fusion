from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rest2_physics_fusion.physics.rest2_clear_sky import add_clear_sky
from rest2_physics_fusion.physics.solar_geometry import SiteConfig, add_solar_geometry, standard_pressure_pa


PHYSICS_FEATURES = [
    "ghi_clear_target",
    "dni_clear_target",
    "dhi_clear_target",
    "mu0_target",
    "apparent_zenith",
    "dni_extra",
    "pressure_pa",
    "pwv_cm",
    "aod700",
    "t_clr_dd",
    "clear_sky_index_last",
    "smart_persistence_ghi",
    "clear_sky_index_current",
    "weather_attenuation_prior",
    "weather_adjusted_clear_sky_ghi",
    "clear_sky_residual_last",
    "clear_sky_gap",
    "ghi_clear_horizon_5min",
    "ghi_clear_horizon_4h",
    "ghi_clear_horizon_1d",
    "mu0_horizon_5min",
    "mu0_horizon_4h",
    "mu0_horizon_1d",
    "weather_adjusted_clear_sky_ghi_horizon_5min",
    "weather_adjusted_clear_sky_ghi_horizon_4h",
    "weather_adjusted_clear_sky_ghi_horizon_1d",
]


@dataclass(frozen=True)
class PhysicsConfig:
    site: SiteConfig
    aod700_default: float = 0.08
    pwat_to_cm: float = 0.1
    sky_index_max: float = 2.0
    clear_sky_backend: str = "auto"


def _numeric_or_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _source_from_availability(series: pd.Series, observed_label: str, missing_label: str = "missing") -> pd.Series:
    return pd.Series(
        np.where(series.notna(), observed_label, missing_label),
        index=series.index,
    )


def _observed_mask(series: pd.Series) -> pd.Series:
    return series.notna().astype(float)


def normalize_weather_columns(df: pd.DataFrame, ghi_column: str, config: PhysicsConfig) -> pd.DataFrame:
    out = df.copy()
    out["input_ghi"] = pd.to_numeric(out[ghi_column], errors="coerce")
    out["input_ghi_source"] = out.get("input_ghi_source", "observed_or_forecast")

    temp_raw = _numeric_or_nan(out, "TEMP")
    wind_speed_raw = _numeric_or_nan(out, "WS")
    wind_dir_raw = _numeric_or_nan(out, "WD")
    precip_raw = _numeric_or_nan(out, "PREC")
    sdwe_raw = _numeric_or_nan(out, "SDWE")
    pwat_raw = _numeric_or_nan(out, "PWAT")

    out["temp_c_source"] = _source_from_availability(temp_raw, "observed_or_joined")
    out["wind_speed_source"] = _source_from_availability(wind_speed_raw, "observed_or_joined")
    out["wind_dir_source"] = _source_from_availability(wind_dir_raw, "observed_or_joined")
    out["precip_source"] = _source_from_availability(precip_raw, "observed_or_joined")
    out["sdwe_source"] = _source_from_availability(sdwe_raw, "observed_or_joined")
    out["pwat_source"] = _source_from_availability(pwat_raw, "observed_or_joined")

    out["temp_c_is_observed"] = _observed_mask(temp_raw)
    out["wind_speed_is_observed"] = _observed_mask(wind_speed_raw)
    out["wind_dir_is_observed"] = _observed_mask(wind_dir_raw)
    out["precip_is_observed"] = _observed_mask(precip_raw)
    out["sdwe_is_observed"] = _observed_mask(sdwe_raw)
    out["pwat_is_observed"] = _observed_mask(pwat_raw)

    out["temp_c"] = temp_raw.fillna(0.0)
    out["wind_speed"] = wind_speed_raw.fillna(0.0)
    out["wind_dir"] = wind_dir_raw.fillna(0.0)
    out["precip"] = precip_raw.fillna(0.0)
    out["sdwe"] = sdwe_raw.fillna(0.0)
    out["pwat_raw"] = pwat_raw
    out["aod700"] = config.aod700_default
    out["pressure_pa"] = standard_pressure_pa(config.site.altitude_m)
    out["pressure_pa_source"] = "altitude_estimated"
    out["pressure_pa_is_observed"] = 0.0
    out["aod700_source"] = "default"
    out["aod700_is_observed"] = 0.0

    pwv_from_pwat = (out["pwat_raw"] * config.pwat_to_cm).clip(lower=0.05, upper=10.0)
    out["pwv_cm"] = pwv_from_pwat.fillna(1.5)
    out["pwv_cm_source"] = pd.Series(
        np.where(out["pwat_raw"].notna(), "PWAT", "default_pwv_1p5cm"),
        index=out.index,
    )
    out["pwv_cm_is_observed"] = out["pwat_raw"].notna().astype(float)
    out["weather_is_joined"] = out.get("weather_is_joined", out["temp_c_is_observed"]).astype(float)
    out["weather_join_status"] = out.get(
        "weather_join_status",
        pd.Series(np.where(out["weather_is_joined"] > 0.5, "available_in_source", "missing"), index=out.index),
    )

    flags = ["pressure_altitude_estimated", f"altitude_{config.site.altitude_m:g}m", "aod700_default"]
    if (out["pwv_cm_is_observed"] < 0.5).any():
        flags.append("some_pwv_default")
    if (out["temp_c_is_observed"] < 0.5).all():
        flags.append("missing_temp")
    if (out["weather_is_joined"] < 0.5).all():
        flags.append("missing_weather")
    out["feature_quality_flags"] = ";".join(flags)
    return out


def add_persistence_features(df: pd.DataFrame, sky_index_max: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    previous_ghi = out["input_ghi"].shift(1)
    previous_clear = out["ghi_clear_target"].shift(1)
    valid = previous_clear > 20.0
    sky_index = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    sky_index.loc[valid] = (previous_ghi.loc[valid] / previous_clear.loc[valid]).clip(
        lower=0.0,
        upper=sky_index_max,
    )
    out["clear_sky_index_last"] = sky_index
    out["smart_persistence_ghi"] = out["clear_sky_index_last"] * out["ghi_clear_target"]
    current_valid = out["ghi_clear_target"] > 20.0
    current_index = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    current_index.loc[current_valid] = (out["input_ghi"].loc[current_valid] / out["ghi_clear_target"].loc[current_valid]).clip(
        lower=0.0,
        upper=sky_index_max,
    )
    out["clear_sky_index_current"] = current_index
    out["clear_sky_residual_last"] = (previous_ghi - previous_clear).fillna(0.0)
    out["clear_sky_gap"] = (out["ghi_clear_target"] - out["input_ghi"]).clip(lower=0.0)
    return out


def add_weather_attenuation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-conditioned clear-sky attenuation priors.

    These are deterministic physical priors for the training model, not final
    predictions. The neural model can learn when to trust or ignore them.
    """

    out = df.copy()
    pwv = pd.to_numeric(out["pwv_cm"], errors="coerce").fillna(1.5).clip(lower=0.05, upper=10.0)
    precip = pd.to_numeric(out["precip"], errors="coerce").fillna(0.0).clip(lower=0.0)
    wind_speed = pd.to_numeric(out["wind_speed"], errors="coerce").fillna(0.0).clip(lower=0.0)
    weather_joined = pd.to_numeric(out["weather_is_joined"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    sky_last = pd.to_numeric(out["clear_sky_index_last"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=2.0)

    water_penalty = 0.025 * np.clip(pwv - 1.5, 0.0, None)
    precip_penalty = 0.18 * (1.0 - np.exp(-precip))
    low_sky_penalty = 0.35 * np.clip(1.0 - sky_last, 0.0, 1.0)
    wind_recovery = 0.03 * np.tanh(wind_speed / 6.0)
    attenuation = 1.0 - water_penalty - precip_penalty - low_sky_penalty + wind_recovery
    attenuation = np.clip(attenuation * (0.7 + 0.3 * weather_joined), 0.05, 1.2)
    out["weather_attenuation_prior"] = attenuation
    out["weather_adjusted_clear_sky_ghi"] = out["ghi_clear_target"] * out["weather_attenuation_prior"]
    return out


def add_horizon_aligned_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    horizons = {
        "5min": 1,
        "4h": 16,
        "1d": 96,
    }
    for name, steps in horizons.items():
        out[f"ghi_clear_horizon_{name}"] = out["ghi_clear_target"].shift(-steps).fillna(out["ghi_clear_target"])
        out[f"mu0_horizon_{name}"] = out["mu0_target"].shift(-steps).fillna(out["mu0_target"])
        out[f"weather_adjusted_clear_sky_ghi_horizon_{name}"] = (
            out["weather_adjusted_clear_sky_ghi"].shift(-steps).fillna(out["weather_adjusted_clear_sky_ghi"])
        )
    return out


def build_physics_features(
    df: pd.DataFrame,
    *,
    timestamp_column: str,
    ghi_column: str,
    config: PhysicsConfig,
    source_type: str,
    station_name: str,
) -> pd.DataFrame:
    if timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")
    if ghi_column not in df.columns:
        raise ValueError(f"Missing GHI column: {ghi_column}")

    out = normalize_weather_columns(df, ghi_column, config)
    out["source_type"] = source_type
    out["station_name"] = station_name
    out["latitude"] = config.site.latitude
    out["longitude"] = config.site.longitude
    out["altitude_m"] = config.site.altitude_m
    out["timezone"] = config.site.timezone
    out = out.sort_values(timestamp_column).reset_index(drop=True)
    out = add_solar_geometry(out, timestamp_column, config.site)
    out = add_clear_sky(out, config.clear_sky_backend)
    out = add_persistence_features(out, config.sky_index_max)
    out = add_weather_attenuation_features(out)
    out = add_horizon_aligned_features(out)
    return out


def require_physics_columns(df: pd.DataFrame) -> None:
    missing = [column for column in PHYSICS_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing physics feature columns: {missing}")
