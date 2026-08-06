from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from pv_physics_moe import IntegratedPVPhysicsMoE, load_config
from pv_physics_moe.data.dataset import collate_solar_batch
from pv_physics_moe.data.station import StationFiveCsvDataset


ROOT = Path(__file__).resolve().parents[1]


def write_five_csvs(root: Path) -> None:
    # Eight days cross the calendar boundary, leaving enough room for 4 h
    # history and the 1-day target on both sides of New Year.
    times = pd.date_range("2024-12-28 00:00:00", periods=768, freq="15min")
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    sun = np.maximum(0.0, np.sin((hour - 6.0) / 12.0 * np.pi))
    ghi, dni, dhi = 850.0 * sun, 700.0 * sun, 150.0 * sun
    power = 0.8 * ghi
    power[120] = -3.0
    pd.DataFrame({"dtime": times, "observe_power": power, "observe_ghi": ghi}).to_csv(root / "site_power_ghi.csv", index=False)
    pd.DataFrame({"dtime": times, "observe_ghi": ghi, "observe_dni": dni, "observe_dhi": dhi}).to_csv(root / "site_irradiance.csv", index=False)
    pd.DataFrame({"dtime": times, "GHI": ghi, "TEMP": 20 + 8 * sun, "WS": 2.0, "WD": 180.0, "PREC": 0.0, "PWAT": 1.5, "SDWE": 0.0}).to_csv(root / "site_weather.csv", index=False)
    for filename, interval in (("forecast_4h.csv", 240), ("forecast_1d.csv", 1440)):
        issue = times - pd.to_timedelta(interval, unit="m")
        pd.DataFrame({
            "dtime": times, "GHI": ghi * 0.95, "TEMP": 20 + 7 * sun, "WS": 2.2,
            "WD": 185.0, "PREC": 0.0, "PWAT": 1.6, "SDWE": 0.0,
            "interval": interval, "interval_day": interval // 1440,
            "report_date": issue.normalize(), "report_time": issue.strftime("%H_%M"),
            "update_time": 0, "govern_flag": 0,
        }).to_csv(root / filename, index=False)


def test_five_csv_server_pipeline_is_two_modal_and_five_horizon(tmp_path: Path) -> None:
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    write_five_csvs(raw)
    config_path = tmp_path / "config.yaml"
    config = yaml.safe_load((ROOT / "configs" / "server_station.yaml").read_text(encoding="utf-8"))
    config["data"].update({
        "root_dir": str(raw), "processed_dir": str(processed), "history_length": 4,
    })
    config["model"].update({
        "hidden_size": 64, "serial_layers": 1, "fusion_heads": 4, "moe_layers": 1,
        "moe_heads": 4, "num_experts": 4, "num_shared_experts": 1,
        "top_k_experts": 2, "pcd_hidden_size": 32, "pcd_layers": 1,
    })
    config["training"].update({"device": "cpu", "num_workers": 0})
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_station_dataset.py"), "--config", str(config_path)], check=True)

    metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["modalities"] == ["serial", "physics"]
    assert metadata["forecast_lead_minutes"] == [15, 30, 60, 240, 1440]
    assert metadata["forecast_horizon"] == 5
    assert metadata["training_years"] == [2024]
    assert metadata["holdout_years"] == [2025]
    assert metadata["normalization_fit_years"] == [2024]
    assert metadata["client_holdout_locked"] is True
    assert metadata["train_validation_embargo_minutes"] == 1440
    assert metadata["station"]["longitude"] == 100.641
    assert metadata["station"]["latitude"] == 29.919
    assert metadata["station"]["altitude_m"] == 4300.0
    assert 59000.0 < metadata["station"]["pressure_pa"] < 60000.0
    assert metadata["pwat_unit"].startswith("cm")
    assert metadata["negative_power_targets_clipped"] > 0
    assert all(metadata["split_samples"][name] > 0 for name in ("train", "val", "test"))

    timestamps = pd.DatetimeIndex(np.load(processed / "timeline" / "timestamps.npy"))
    history, max_offset = metadata["history_length"], 96
    for split in ("train", "val"):
        indices = np.load(processed / "splits" / f"{split}_end_indices.npy")
        assert set(timestamps[indices - history + 1].year) == {2024}
        assert set(timestamps[indices].year) == {2024}
        assert set(timestamps[indices + max_offset].year) == {2024}
    test_indices = np.load(processed / "splits" / "test_end_indices.npy")
    assert set(timestamps[test_indices - history + 1].year) == {2025}
    assert set(timestamps[test_indices].year) == {2025}
    assert set(timestamps[test_indices + max_offset].year) == {2025}
    train_indices = np.load(processed / "splits" / "train_end_indices.npy")
    val_indices = np.load(processed / "splits" / "val_end_indices.npy")
    assert int(train_indices[-1]) + max_offset < int(val_indices[0])

    normalizers = json.loads((processed / "normalization.json").read_text(encoding="utf-8"))
    assert normalizers["serial"]["fit_years"] == [2024]
    assert normalizers["physics"]["fit_years"] == [2024]

    dataset = StationFiveCsvDataset(processed, "train", limit=2)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_solar_batch)))
    assert batch["serial"].shape == (2, 4, 13)
    assert batch["physics"].shape == (2, 5, 26)
    assert batch["target"].shape == (2, 5, 1)
    assert batch["irradiance_target"].shape == (2, 5, 3)
    assert batch["future_zenith"].shape == (2, 5)
    assert "images" not in batch

    model = IntegratedPVPhysicsMoE(load_config(config_path)).eval()
    assert model.image_encoder is None
    with torch.no_grad():
        output = model(batch)
    assert output["prediction"].shape == (2, 5, 1)
    assert output["irradiance"].shape == (2, 5, 3)
    assert set(output["fusion_output"].modality_embeds) == {"serial", "physics"}
    assert set(output["fusion_output"].attention_maps) == {"serial_from_physics", "physics_from_serial"}
    ghi_out, dni_out, dhi_out = output["irradiance"].unbind(-1)
    closure = torch.max(torch.abs(ghi_out - output["future_cos_zenith"].clamp_min(0) * dni_out - dhi_out))
    assert closure.item() < 1e-3
