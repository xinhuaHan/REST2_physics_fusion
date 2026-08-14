from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pv_physics_moe import IntegratedPVPhysicsMoE, load_parquet_config
from pv_physics_moe.data import ConfigurableParquetDataset, build_parquet_datasets, collate_parquet_batch
from pv_physics_moe.evaluation import finalize_prediction_frame, official_metrics, prediction_records
from pv_physics_moe.training.losses import IntegratedLoss


ROOT = Path(__file__).resolve().parents[1]


def synthetic_frame(config, rows: int, start: str = "2024-01-01") -> pd.DataFrame:
    cfg = config.dataset
    times = pd.date_range(start, periods=rows, freq=f"{cfg.sampling_interval_minutes}min")
    data = {cfg.timestamp_column: times}
    for index, column in enumerate(cfg.serial_columns):
        data[column] = np.arange(rows, dtype=np.float32) + index + 1
    data[cfg.target_column] = np.arange(rows, dtype=np.float32) + 10
    for column in cfg.irradiance_columns.values():
        if column:
            values = np.maximum(0, np.sin(np.arange(rows) / 10.0) * 500).astype(np.float32)
            timestamp_column = cfg.list_timestamp_columns.get(column)
            if timestamp_column:
                data[column] = [[float(value)] * 5 for value in values]
                data[timestamp_column] = [
                    [time - pd.Timedelta(minutes=offset) for offset in range(4, -1, -1)]
                    for time in times
                ]
            else:
                data[column] = values
    if cfg.name == "YLJ":
        data["GHI-NWP_observe"] = np.arange(rows, dtype=np.float32) + 1
    if cfg.pressure_column:
        data[cfg.pressure_column] = np.full(rows, 101325.0, np.float32)
    if cfg.precip_column:
        data[cfg.precip_column] = np.zeros(rows, np.float32)
    return pd.DataFrame(data)


def configure_synthetic(config, path: Path, start: str, end: str) -> None:
    config.dataset.parquet_file = str(path)
    config.site.latitude = 34.6
    config.site.longitude = 112.4
    config.site.altitude_m = 150.0
    config.site.timezone = "Asia/Shanghai"
    config.splits = {"train": [start, end], "val": [start, end], "test": [start, end]}


def test_configs_derive_field_dimensions_and_forecast_steps():
    ylj = load_parquet_config(ROOT / "configs" / "ylj_parquet.yaml")
    luoyang = load_parquet_config(ROOT / "configs" / "luoyang_parquet.yaml")
    assert (ylj.model.serial_input_dim, ylj.model.forecast_horizon) == (22, 16)
    assert ylj.dataset.parquet_file.endswith("YLJ-Unified_format-with_DNI_DHI.parquet")
    assert ylj.dataset.irradiance_columns == {
        "ghi": "observe_ghi-onsite", "dni": "estimated_DNI-onsite", "dhi": "estimated_DHI-onsite"
    }
    assert ylj.dataset.irradiance_sources == {"ghi": "observed", "dni": "estimated", "dhi": "estimated"}
    assert ylj.dataset.pwv_column == "PWAT-NWP_observe"
    assert ylj.dataset.pwv_unit == "mm"
    assert ylj.dataset.pwv_to_cm == pytest.approx(0.1)
    assert ylj.runtime.precision == "fp32"
    assert ylj.normalization.power_scale == pytest.approx(468.0)
    assert ylj.normalization.rated_power == pytest.approx(468.0)
    assert ylj.evaluation.nrmse_denominator == pytest.approx(468.0)
    assert (luoyang.model.serial_input_dim, luoyang.model.forecast_horizon) == (9, 48)
    changed = copy.deepcopy(ylj)
    changed.dataset.sampling_interval_minutes = 5
    changed.dataset.forecast_step_minutes = 5
    changed.dataset.forecast_steps = None
    changed.validate()
    assert changed.model.forecast_horizon == 48


def test_synthetic_ylj_parquet_is_16_steps_with_dni_dhi_and_pwat_conversion(tmp_path: Path):
    config = load_parquet_config(ROOT / "configs" / "ylj_parquet.yaml")
    frame = synthetic_frame(config, 64)
    path = tmp_path / "ylj.parquet"
    frame.to_parquet(path)
    configure_synthetic(config, path, "2024-01-01", "2024-01-02")
    datasets = build_parquet_datasets(config)
    batch = collate_parquet_batch([datasets["train"][0], datasets["train"][1]])
    assert batch["serial"].shape == (2, 16, 22)
    assert batch["physics"].shape == (2, 16, 26)
    assert batch["target"].shape == (2, 16, 1)
    assert torch.all(batch["irradiance_target_mask"] == 1)
    horizons = [int((pd.Timestamp(t) - pd.Timestamp(batch["issue_time"][0])).total_seconds() / 60) for t in batch["target_times"][0]]
    assert horizons == list(range(15, 241, 15))
    first_issue = datasets["train"].issue_indices[0]
    raw_pwat_mm = float(frame.loc[first_issue, "PWAT-NWP_observe"])
    assert torch.allclose(batch["physics_raw"][0, :, 7], torch.full((16,), raw_pwat_mm * 0.1))

    missing = copy.deepcopy(config)
    missing.dataset.irradiance_columns["dni"] = None
    missing.dataset.irradiance_columns["dhi"] = None
    missing_supervision = ConfigurableParquetDataset(missing, "train", frame=frame)[0]
    assert torch.all(missing_supervision["irradiance_target_mask"][:, 0] == 1)
    assert torch.all(missing_supervision["irradiance_target_mask"][:, 1:] == 0)


def test_ylj_target_floor_clips_negative_power_but_rejects_missing_targets():
    config = load_parquet_config(ROOT / "configs" / "ylj_parquet.yaml")
    frame = synthetic_frame(config, 64)
    frame.loc[16, config.dataset.target_column] = -1.5
    frame.loc[32, config.dataset.target_column] = np.nan
    dataset = ConfigurableParquetDataset(config, "train", frame=frame)
    assert dataset.metadata["target_floor"] == 0.0
    assert dataset.metadata["target_floor_clipped_count"] == 1
    assert dataset[0]["target"][0, 0] == 0.0
    assert all(32 not in dataset._future_indices(issue) for issue in dataset.issue_indices)


def test_luoyang_parquet_images_offsets_masks_and_48_targets(tmp_path: Path):
    config = load_parquet_config(ROOT / "configs" / "luoyang_parquet.yaml")
    frame = synthetic_frame(config, 90, "2026-04-05")
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.fromarray(np.full((8, 8, 3), 127, np.uint8)).save(image_root / "sky.png")
    paths, stamps = [], []
    for time in frame["timestamp"]:
        paths.append(["sky.png", "missing.png"])
        stamps.append([(time - pd.Timedelta(minutes=10)).isoformat(), time.isoformat()])
    frame[config.images.paths_column] = paths
    frame[config.images.timestamps_column] = stamps
    path = tmp_path / "luoyang.parquet"
    frame.to_parquet(path)
    configure_synthetic(config, path, "2026-04-05", "2026-04-06")
    config.images.root = str(image_root)
    dataset = ConfigurableParquetDataset(config, "train")
    sample = dataset[0]
    assert sample["target"].shape == (48, 1)
    assert sample["images"].shape == (16, 3, 64, 64)
    assert sample["image_valid_mask"].sum() == 1
    assert -75 <= float(sample["image_time_offsets"][0]) <= 0
    assert pd.Timestamp(sample["target_times"][-1]) - pd.Timestamp(sample["issue_time"]) == pd.Timedelta(minutes=240)


def test_luoyang_list_irradiance_matches_timestamp_not_list_position():
    config = load_parquet_config(ROOT / "configs" / "luoyang_parquet.yaml")
    frame = synthetic_frame(config, 90, "2026-04-05")
    frame[config.images.paths_column] = [[] for _ in range(len(frame))]
    frame[config.images.timestamps_column] = [[] for _ in range(len(frame))]
    first_future_row = config.dataset.history_points
    target_time = frame.loc[first_future_row, "timestamp"]
    for component, expected in (("ghi", 601.0), ("dni", 501.0), ("dhi", 101.0)):
        column = config.dataset.irradiance_columns[component]
        timestamp_column = config.dataset.list_timestamp_columns[column]
        frame.at[first_future_row, column] = [expected, -999.0]
        frame.at[first_future_row, timestamp_column] = [
            target_time, target_time - pd.Timedelta(minutes=1)
        ]
    config.splits = {"train": ["2026-04-05", "2026-04-06"]}
    dataset = ConfigurableParquetDataset(config, "train", frame=frame)
    sample = dataset[0]
    assert torch.equal(sample["irradiance_target"][0], torch.tensor([601.0, 501.0, 101.0]))
    assert torch.equal(sample["irradiance_target_mask"][0], torch.ones(3, dtype=torch.long))


def test_rejects_bad_timestamps_and_image_metadata(tmp_path: Path):
    config = load_parquet_config(ROOT / "configs" / "ylj_parquet.yaml")
    frame = synthetic_frame(config, 40)
    frame.loc[2:, "timestamp"] += pd.Timedelta(minutes=1)
    configure_synthetic(config, tmp_path / "bad.parquet", "2024-01-01", "2024-01-02")
    with pytest.raises(ValueError, match="expected 15 minutes, actual 16 minutes"):
        ConfigurableParquetDataset(config, "train", frame=frame)
    duplicate = synthetic_frame(config, 40)
    duplicate.loc[2, "timestamp"] = duplicate.loc[1, "timestamp"]
    with pytest.raises(ValueError, match="duplicate timestamp"):
        ConfigurableParquetDataset(config, "train", frame=duplicate)

    luoyang = load_parquet_config(ROOT / "configs" / "luoyang_parquet.yaml")
    image_frame = synthetic_frame(luoyang, 80, "2026-04-05")
    image_frame[luoyang.images.paths_column] = [["x.png"] for _ in range(len(image_frame))]
    image_frame[luoyang.images.timestamps_column] = [[(time + pd.Timedelta(minutes=1)).isoformat()] for time in image_frame["timestamp"]]
    configure_synthetic(luoyang, tmp_path / "future.parquet", "2026-04-05", "2026-04-06")
    luoyang.images.root = str(tmp_path)
    dataset = ConfigurableParquetDataset(luoyang, "train", frame=image_frame)
    with pytest.raises(ValueError, match="future image"):
        dataset[0]

    mismatch_frame = synthetic_frame(luoyang, 80, "2026-04-05")
    mismatch_frame[luoyang.images.paths_column] = [
        ["a.png", "b.png"] for _ in range(len(mismatch_frame))
    ]
    mismatch_frame[luoyang.images.timestamps_column] = [
        [time.isoformat()] for time in mismatch_frame["timestamp"]
    ]
    mismatch_dataset = ConfigurableParquetDataset(luoyang, "train", frame=mismatch_frame)
    with pytest.raises(ValueError, match="length mismatch"):
        mismatch_dataset[0]


def test_training_normalization_never_reads_validation_values():
    config = load_parquet_config(ROOT / "configs" / "ylj_parquet.yaml")
    frame = synthetic_frame(config, 96)
    frame.loc[64:, config.dataset.serial_columns] = 1_000_000.0
    config.site.timezone = "Asia/Shanghai"
    config.splits = {
        "train": ["2024-01-01", "2024-01-01 16:00:00"],
        "val": ["2024-01-01 16:00:00", "2024-01-02"],
    }
    train = ConfigurableParquetDataset(config, "train", frame=frame)
    assert float(train.normalization.serial_mean.max()) < 1000.0
    val = ConfigurableParquetDataset(config, "val", normalization=train.normalization, frame=frame)
    assert val.normalization is train.normalization


def test_irradiance_loss_is_finite_with_no_valid_dni_dhi():
    config = load_parquet_config(ROOT / "configs" / "ylj_parquet.yaml")
    config.model.hidden_size = 32
    config.model.serial_layers = 1
    config.model.fusion_heads = 4
    config.model.moe_layers = 1
    config.model.moe_heads = 4
    config.model.num_experts = 2
    config.model.num_shared_experts = 1
    config.model.top_k_experts = 1
    config.model.pcd_hidden_size = 16
    config.model.pcd_layers = 1
    model = IntegratedPVPhysicsMoE(config.model)
    batch = {
        "serial": torch.rand(2, 16, 22),
        "physics": torch.rand(2, 16, 26),
        "physics_raw": torch.rand(2, 16, 26),
        "future_zenith": torch.full((2, 16), 30.0),
        "target": torch.rand(2, 16, 1),
        "target_valid_mask": torch.ones(2, 16),
        "irradiance_target": torch.zeros(2, 16, 3),
        "irradiance_target_mask": torch.zeros(2, 16, 3),
    }
    output = model(batch)
    losses = IntegratedLoss(config.training)(output, batch)
    assert losses["irradiance_loss"] == 0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_integrated_loss_uses_fp32_reduction_for_ylj_scale_targets():
    config = load_parquet_config(ROOT / "configs" / "ylj_parquet.yaml")
    outputs = {
        "prediction": torch.zeros(2, 16, 1, dtype=torch.float16, requires_grad=True),
        "irradiance": torch.zeros(2, 16, 3, dtype=torch.float16, requires_grad=True),
        "raw_irradiance": torch.zeros(2, 16, 3, dtype=torch.float16, requires_grad=True),
        "clear_sky_prior": torch.full((2, 16, 3), 500.0, dtype=torch.float16),
        "moe_balance_loss": torch.zeros((), dtype=torch.float16, requires_grad=True),
    }
    batch = {
        "target": torch.full((2, 16, 1), 468.0, dtype=torch.float16),
        "target_valid_mask": torch.ones(2, 16),
        "irradiance_target": torch.full((2, 16, 3), 1_000.0, dtype=torch.float16),
        "irradiance_target_mask": torch.ones(2, 16, 3),
    }
    losses = IntegratedLoss(config.training)(outputs, batch)
    assert losses["loss"].dtype == torch.float32
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_prediction_export_counts_and_metrics_select_horizon_minutes():
    evaluation = load_parquet_config(ROOT / "configs" / "luoyang_parquet.yaml").evaluation
    batch = {
        "issue_time": ["2026-04-05T00:00:00"],
        "target_times": [[(pd.Timestamp("2026-04-05") + pd.Timedelta(minutes=5 * step)).isoformat() for step in range(1, 49)]],
        "target": torch.arange(48, dtype=torch.float32).view(1, 48, 1),
        "current_power": torch.tensor([[0.0]]),
    }
    prediction = batch["target"] + 10.0
    records = prediction_records(batch, prediction)
    frame = finalize_prediction_frame(records, 48)
    assert len(frame) == 48
    metrics = official_metrics(frame, evaluation)
    assert set(metrics["horizons_minutes"]) == {"15", "240"}
    assert metrics["horizons_minutes"]["15"]["overall"]["RMSE"] == pytest.approx(10.0)
    assert metrics["horizons_minutes"]["240"]["overall"]["NMAE"] == pytest.approx(10.0 / 48629.73)

    ylj_times = [[(pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=15 * step)).isoformat() for step in range(1, 17)]]
    ylj_batch = {
        **batch, "issue_time": ["2024-01-01T00:00:00"],
        "target_times": ylj_times, "target": torch.zeros(1, 16, 1),
    }
    assert len(finalize_prediction_frame(prediction_records(ylj_batch, torch.zeros(1, 16, 1)), 16)) == 16
