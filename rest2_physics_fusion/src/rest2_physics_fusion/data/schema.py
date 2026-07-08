from __future__ import annotations

from dataclasses import dataclass, field

from rest2_physics_fusion.physics.physics_features import PHYSICS_FEATURES


@dataclass(frozen=True)
class DataSchema:
    timestamp_column: str = "timestamp"
    target_column: str = "target_ghi_5min"
    auxiliary_target_column: str | None = None
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


def infer_auxiliary_ghi_target(target_column: str) -> str | None:
    mapping = {
        "target_power_5min": "target_ghi_5min",
        "target_power_4h": "target_ghi_4h",
        "target_power_1d": "target_ghi_1d",
    }
    return mapping.get(target_column)


def schema_from_config(data_cfg: dict, target_column: str | None = None) -> DataSchema:
    base = DataSchema()
    resolved_target = target_column or data_cfg.get("target_column", base.target_column)
    serial_columns = tuple(data_cfg.get("feature_columns", base.serial_feature_columns))
    weather_columns = tuple(data_cfg.get("weather_feature_columns", base.weather_feature_columns))
    physics_columns = tuple(data_cfg.get("physics_features", base.physics_feature_columns))
    auxiliary_target = data_cfg.get("auxiliary_target_column") or infer_auxiliary_ghi_target(str(resolved_target))
    return DataSchema(
        timestamp_column=str(data_cfg.get("timestamp_column", base.timestamp_column)),
        target_column=str(resolved_target),
        auxiliary_target_column=auxiliary_target,
        serial_feature_columns=serial_columns,
        weather_feature_columns=weather_columns,
        physics_feature_columns=physics_columns,
    )


def schema_from_state(schema_state: dict, target_column: str | None = None) -> DataSchema:
    base = DataSchema()
    resolved_target = target_column or schema_state.get("target_column", base.target_column)
    auxiliary_target = (
        infer_auxiliary_ghi_target(str(resolved_target))
        if target_column is not None
        else schema_state.get("auxiliary_target_column") or infer_auxiliary_ghi_target(str(resolved_target))
    )
    return DataSchema(
        timestamp_column=str(schema_state.get("timestamp_column", base.timestamp_column)),
        target_column=str(resolved_target),
        auxiliary_target_column=auxiliary_target,
        serial_feature_columns=tuple(schema_state.get("serial_feature_columns", base.serial_feature_columns)),
        weather_feature_columns=tuple(schema_state.get("weather_feature_columns", base.weather_feature_columns)),
        physics_feature_columns=tuple(schema_state.get("physics_feature_columns", base.physics_feature_columns)),
    )


def validate_columns(columns: set[str], schema: DataSchema) -> None:
    required = {
        schema.timestamp_column,
        schema.target_column,
        *schema.serial_feature_columns,
        *schema.weather_feature_columns,
        *schema.physics_feature_columns,
    }
    if schema.auxiliary_target_column:
        required.add(schema.auxiliary_target_column)
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
