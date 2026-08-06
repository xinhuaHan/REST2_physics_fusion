from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict
import os
import re

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_environment(value: Any) -> Any:
    """Expand ${NAME} and ${NAME:-default} recursively in YAML values."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ValueError(f"environment variable {name!r} is required by the configuration")
        return os.path.expanduser(_ENV_PATTERN.sub(replace, value))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


@dataclass
class ModelConfig:
    serial_input_dim: int = 13
    physics_input_dim: int = 26
    use_image: bool = False
    image_channels: int = 3
    image_size: int = 64
    image_cnn_width: int = 32
    image_temporal_layers: int = 1
    image_temporal_heads: int = 8
    image_time_scale_minutes: float = 5.0
    hidden_size: int = 768
    serial_layers: int = 2
    fusion_heads: int = 8
    moe_layers: int = 4
    moe_heads: int = 8
    num_experts: int = 8
    num_shared_experts: int = 2
    top_k_experts: int = 2
    pcd_hidden_size: int = 128
    pcd_layers: int = 2
    forecast_horizon: int = 5
    target_mode: str = "power"
    dropout: float = 0.1
    raw_state_residual_scale: float = 500.0
    night_cos_threshold: float = 0.0

    def validate(self) -> None:
        if self.target_mode not in {"power", "irradiance"}:
            raise ValueError("model.target_mode must be 'power' or 'irradiance'")
        if self.hidden_size % self.fusion_heads or self.hidden_size % self.moe_heads:
            raise ValueError("hidden_size must be divisible by fusion_heads and moe_heads")
        if self.use_image and self.hidden_size % self.image_temporal_heads:
            raise ValueError("hidden_size must be divisible by image_temporal_heads")
        if self.image_channels <= 0 or self.image_size <= 0 or self.image_cnn_width <= 0:
            raise ValueError("image_channels, image_size and image_cnn_width must be positive")
        if self.image_temporal_layers < 1 or self.image_temporal_heads < 1:
            raise ValueError("image temporal layers and heads must be positive")
        if self.image_time_scale_minutes <= 0:
            raise ValueError("image_time_scale_minutes must be positive")
        if not 1 <= self.top_k_experts <= self.num_experts:
            raise ValueError("top_k_experts must be in [1, num_experts]")
        if self.physics_input_dim != 26:
            raise ValueError("the REST2 feature contract currently contains exactly 26 features")


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    task_loss_weight: float = 1.0
    irradiance_loss_weight: float = 0.2
    moe_balance_weight: float = 0.01
    raw_correction_weight: float = 1e-6
    gradient_clip: float = 1.0
    num_workers: int = 0
    device: str = "cpu"
    output_dir: str = "outputs/station"


@dataclass
class DataConfig:
    """All paths and station-specific data assumptions live in this one section."""
    root_dir: str = "${STATION_DATA_ROOT:-data/raw/station}"
    processed_dir: str = "${STATION_PROCESSED_DIR:-data/processed/station}"
    power_ghi_file: str = "site_power_ghi.csv"
    irradiance_file: str = "site_irradiance.csv"
    forecast_4h_file: str = "forecast_4h.csv"
    forecast_1d_file: str = "forecast_1d.csv"
    weather_file: str = "site_weather.csv"
    timestamp_column: str = "dtime"
    power_column: str = "observe_power"
    power_ghi_column: str = "observe_ghi"
    irradiance_ghi_column: str = "observe_ghi"
    irradiance_dni_column: str = "observe_dni"
    irradiance_dhi_column: str = "observe_dhi"
    weather_columns: list[str] = field(default_factory=lambda: ["GHI", "TEMP", "WS", "WD", "PREC", "PWAT", "SDWE"])
    forecast_columns: list[str] = field(default_factory=lambda: ["GHI", "TEMP", "WS", "WD", "PREC", "PWAT", "SDWE"])
    forecast_issue_date_column: str = "report_date"
    forecast_issue_time_column: str = "report_time"
    forecast_interval_column: str = "interval"
    latitude: float = 29.919
    longitude: float = 100.641
    altitude_m: float = 4300.0
    timezone: str = "Asia/Shanghai"
    sample_minutes: int = 15
    history_length: int = 16
    forecast_lead_minutes: list[int] = field(default_factory=lambda: [15, 30, 60, 240, 1440])
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    power_target_floor: float = 0.0
    default_pressure_pa: float | None = None
    default_aod700: float = 0.08

    def validate(self) -> None:
        if self.sample_minutes <= 0 or self.history_length <= 0:
            raise ValueError("sample_minutes and history_length must be positive")
        if not self.forecast_lead_minutes or any(value <= 0 for value in self.forecast_lead_minutes):
            raise ValueError("forecast_lead_minutes must contain positive values")
        if sorted(set(self.forecast_lead_minutes)) != self.forecast_lead_minutes:
            raise ValueError("forecast_lead_minutes must be strictly increasing and unique")
        if any(value % self.sample_minutes for value in self.forecast_lead_minutes):
            raise ValueError("every forecast lead must be an integer multiple of sample_minutes")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("data.train_fraction must be in (0,1)")
        if not 0.0 < self.val_fraction < 1.0 or self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("data.val_fraction must be positive and leave a non-empty test fraction")
        if not -90.0 <= self.latitude <= 90.0 or not -180.0 <= self.longitude <= 180.0:
            raise ValueError("invalid station latitude/longitude")
        if self.altitude_m < -500.0 or self.altitude_m > 10000.0:
            raise ValueError("data.altitude_m is outside the supported standard-atmosphere range")
        if self.default_pressure_pa is not None and self.default_pressure_pa <= 0:
            raise ValueError("data.default_pressure_pa must be positive or null")


@dataclass
class ProjectConfig:
    seed: int = 42


@dataclass
class IntegratedConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        if self.model.forecast_horizon != len(self.data.forecast_lead_minutes):
            raise ValueError("model.forecast_horizon must equal len(data.forecast_lead_minutes)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> IntegratedConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = _expand_environment(yaml.safe_load(handle) or {})
    config = IntegratedConfig(
        project=ProjectConfig(**raw.get("project", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
    )
    config.validate()
    return config
