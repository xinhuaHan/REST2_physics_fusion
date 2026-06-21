from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ModelVariant:
    name: str
    use_rest2_calibration: bool
    use_weather_prior_fusion: bool
    use_clear_sky_weather_head: bool = True
    use_clear_sky_power_prior: bool = False
    weather_prior_weight_max: float = 1.0


MODEL_VARIANTS: dict[str, ModelVariant] = {
    "baseline": ModelVariant("baseline", False, False),
    "physics_feature_only": ModelVariant("physics_feature_only", False, False, use_clear_sky_weather_head=False),
    "clear_sky_power_prior": ModelVariant("clear_sky_power_prior", False, False, use_clear_sky_power_prior=True),
    "weather_prior_weak": ModelVariant("weather_prior_weak", False, True, weather_prior_weight_max=0.25),
    "rest2_calibrated": ModelVariant("rest2_calibrated", True, False),
    "weather_prior": ModelVariant("weather_prior", False, True),
    "weather_prior_rest2": ModelVariant("weather_prior_rest2", True, True),
}


def dataset_key(path: str | Path) -> str:
    stem = Path(path).stem
    if stem.startswith("train_"):
        return stem[len("train_") :]
    return stem


def resolve_model_variant(name: str) -> ModelVariant:
    if name not in MODEL_VARIANTS:
        valid = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unknown model_type={name!r}. Valid values: {valid}")
    return MODEL_VARIANTS[name]


def select_model_type(
    selection_csv: str | Path,
    *,
    csv_path: str | Path,
    target_column: str,
    default_model_type: str = "baseline",
) -> str:
    table = pd.read_csv(selection_csv)
    required = {"dataset", "target_column", "selected_model_type"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Model selection table is missing columns: {missing}")

    dataset_name = Path(csv_path).name
    key = dataset_key(csv_path)
    matches = table[
        (table["target_column"].astype(str) == str(target_column))
        & (
            (table["dataset"].astype(str) == dataset_name)
            | (table["dataset"].astype(str) == key)
            | (table["dataset"].astype(str) == Path(dataset_name).stem)
        )
    ]
    if matches.empty:
        return default_model_type
    selected = str(matches.iloc[0]["selected_model_type"])
    resolve_model_variant(selected)
    return selected
