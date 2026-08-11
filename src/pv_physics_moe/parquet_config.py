from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import ModelConfig, ProjectConfig, TrainingConfig, _expand_environment


@dataclass
class ParquetDatasetConfig:
    name: str
    parquet_file: str | None
    timestamp_column: str
    target_column: str
    serial_columns: list[str]
    irradiance_columns: dict[str, str | None]
    sampling_interval_minutes: int
    history_points: int = 16
    forecast_step_minutes: int = 15
    forecast_steps: int | None = None
    forecast_horizon_minutes: int = 240
    causal_fill: str = "forward_fill"
    pressure_column: str | None = None
    pressure_unit: str = "pa"
    wind_columns: list[str] = field(default_factory=list)
    precip_column: str | None = None
    pwv_column: str | None = None
    pwv_unit: str | None = None
    pwv_to_cm: float | None = None

    def validate(self) -> None:
        if not self.serial_columns or len(set(self.serial_columns)) != len(self.serial_columns):
            raise ValueError("dataset.serial_columns must be non-empty and unique")
        if self.sampling_interval_minutes <= 0 or self.forecast_step_minutes <= 0:
            raise ValueError("sampling and forecast step minutes must be positive")
        if self.forecast_horizon_minutes != 240:
            raise ValueError("forecast_horizon_minutes must be 240 for this task")
        if self.forecast_horizon_minutes % self.forecast_step_minutes:
            raise ValueError("240 must be divisible by dataset.forecast_step_minutes")
        derived = self.forecast_horizon_minutes // self.forecast_step_minutes
        if self.forecast_steps is not None and self.forecast_steps != derived:
            raise ValueError(
                f"dataset.forecast_steps={self.forecast_steps} does not match "
                f"240/forecast_step_minutes={derived}"
            )
        self.forecast_steps = derived
        if self.history_points <= 0:
            raise ValueError("dataset.history_points must be positive")
        if self.causal_fill not in {"forward_fill", "none"}:
            raise ValueError("dataset.causal_fill must be forward_fill or none")
        if self.pwv_column and (self.pwv_unit is None or self.pwv_to_cm is None):
            raise ValueError(
                "dataset.pwv_unit and dataset.pwv_to_cm are required before a PWV/PWAT column can be used"
            )
        expected_pwv_factors = {"mm": 0.1, "cm": 1.0, "kg_m2": 0.1}
        if self.pwv_unit is not None:
            if self.pwv_unit not in expected_pwv_factors:
                raise ValueError("dataset.pwv_unit must be one of: mm, cm, kg_m2")
            expected_factor = expected_pwv_factors[self.pwv_unit]
            if self.pwv_to_cm is None or abs(self.pwv_to_cm - expected_factor) > 1e-12:
                raise ValueError(
                    f"dataset.pwv_unit={self.pwv_unit} requires pwv_to_cm={expected_factor}"
                )


@dataclass
class SiteConfig:
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    timezone: str | None = None

    def require_solar_geometry(self) -> None:
        missing = [name for name in ("latitude", "longitude", "timezone") if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "solar geometry cannot be built; missing site parameters: " + ", ".join(missing)
            )


@dataclass
class PhysicsDefaultsConfig:
    pressure_pa: float = 101325.0
    pwv_cm: float = 1.5
    aod700: float = 0.08
    precip: float = 0.0


@dataclass
class ImagesConfig:
    enabled: bool = False
    root: str | None = None
    paths_column: str | None = None
    timestamps_column: str | None = None
    max_frames: int = 16
    channels: int = 3
    size: int = 64
    sampling_strategy: str = "uniform"
    reject_future: bool = True
    history_minutes: int = 75

    def validate(self) -> None:
        if self.max_frames <= 0 or self.channels <= 0 or self.size <= 0:
            raise ValueError("image dimensions and max_frames must be positive")
        if self.sampling_strategy != "uniform":
            raise ValueError("only images.sampling_strategy=uniform is supported")
        if self.enabled:
            missing = [name for name in ("root", "paths_column", "timestamps_column") if not getattr(self, name)]
            if missing:
                raise ValueError("enabled images are missing configuration: " + ", ".join(missing))


@dataclass
class NormalizationConfig:
    power_scale: float | None = None
    rated_power: float | None = None


@dataclass
class EvaluationConfig:
    horizons_minutes: list[int] = field(default_factory=lambda: [15, 240])
    nrmse_denominator: float | None = None
    hard_delta_fraction: float = 0.10
    direction_min_delta_fraction: float = 0.02

    def validate(self) -> None:
        if self.horizons_minutes != [15, 240]:
            raise ValueError("official evaluation horizons must be exactly [15, 240]")
        if self.nrmse_denominator is not None and self.nrmse_denominator <= 0:
            raise ValueError("evaluation.nrmse_denominator must be positive or null")


@dataclass
class RuntimeConfig:
    distributed: bool = True
    gpu_devices: str = "0,1,2,3,4,5,6,7"
    precision: str = "fp16"
    num_workers: int = 4
    pin_memory: bool = True

    def validate(self) -> None:
        if self.precision not in {"fp16", "fp32"}:
            raise ValueError("runtime.precision must be fp16 or fp32; V100 does not support bf16")


@dataclass
class ParquetForecastConfig:
    dataset: ParquetDatasetConfig
    site: SiteConfig
    physics_defaults: PhysicsDefaultsConfig
    images: ImagesConfig
    normalization: NormalizationConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig
    splits: dict[str, Any]
    model: ModelConfig
    training: TrainingConfig
    project: ProjectConfig = field(default_factory=ProjectConfig)

    def validate(self) -> None:
        self.dataset.validate()
        self.images.validate()
        self.evaluation.validate()
        self.runtime.validate()
        unavailable = [
            horizon for horizon in self.evaluation.horizons_minutes
            if horizon % self.dataset.forecast_step_minutes
        ]
        if unavailable:
            raise ValueError(
                "forecast_step_minutes cannot produce required official horizon_minutes: "
                f"{unavailable}"
            )
        self.model.serial_input_dim = len(self.dataset.serial_columns)
        self.model.forecast_horizon = int(self.dataset.forecast_steps)
        self.model.use_image = self.images.enabled
        self.model.image_channels = self.images.channels
        self.model.image_size = self.images.size
        self.model.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_parquet_config(path: str | Path) -> ParquetForecastConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = _expand_environment(yaml.safe_load(handle) or {})
    required = {"dataset", "site", "physics_defaults", "images", "normalization", "evaluation", "runtime"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"parquet config is missing sections: {sorted(missing)}")
    config = ParquetForecastConfig(
        dataset=ParquetDatasetConfig(**raw["dataset"]),
        site=SiteConfig(**raw["site"]),
        physics_defaults=PhysicsDefaultsConfig(**raw["physics_defaults"]),
        images=ImagesConfig(**raw["images"]),
        normalization=NormalizationConfig(**raw["normalization"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        splits=raw.get("splits", {}),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        project=ProjectConfig(**raw.get("project", {})),
    )
    config.validate()
    return config
