from __future__ import annotations

from dataclasses import dataclass, field

from rest2_physics_fusion.physics.physics_features import PHYSICS_FEATURES


@dataclass(frozen=True)
class DataSchema:
    timestamp_column: str = "dtime"
    target_column: str = "target_ghi_5min"
    serial_feature_columns: tuple[str, ...] = (
        "input_ghi",
        "temp_c",
        "wind_speed",
        "wind_dir",
        "precip",
        "pwv_cm",
        "temp_c_is_observed",
        "wind_speed_is_observed",
        "wind_dir_is_observed",
        "precip_is_observed",
        "pwv_cm_is_observed",
        "pressure_pa_is_observed",
        "aod700_is_observed",
        "weather_is_joined",
    )
    weather_feature_columns: tuple[str, ...] = (
        "temp_c",
        "wind_speed",
        "wind_dir",
        "precip",
        "pwv_cm",
        "temp_c_is_observed",
        "wind_speed_is_observed",
        "wind_dir_is_observed",
        "precip_is_observed",
        "pwv_cm_is_observed",
        "weather_is_joined",
    )
    physics_feature_columns: tuple[str, ...] = field(default_factory=lambda: tuple(PHYSICS_FEATURES))


def validate_columns(columns: set[str], schema: DataSchema) -> None:
    required = {
        schema.timestamp_column,
        schema.target_column,
        *schema.serial_feature_columns,
        *schema.weather_feature_columns,
        *schema.physics_feature_columns,
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
