from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.data.preprocess import read_training_csv
from rest2_physics_fusion.data.schema import DataSchema
from rest2_physics_fusion.training.losses import regression_metrics
from rest2_physics_fusion.training.train import _deserialize_norm, _load_checkpoint, _serialize_norm, build_loaders
from rest2_physics_fusion.models.serial_physics_model import SerialPhysicsForecaster


DIAGNOSTIC_COLUMNS = [
    "source_type",
    "station_name",
    "input_ghi",
    "ghi_clear_target",
    "clear_sky_index_current",
    "clear_sky_index_last",
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
    "temp_c",
    "wind_speed",
    "precip",
    "pwv_cm",
    "weather_is_joined",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--window-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def schema_from_checkpoint(checkpoint: dict, target_column: str | None) -> DataSchema:
    schema_state = checkpoint.get("schema", {})
    return DataSchema(target_column=target_column or schema_state.get("target_column", "target_ghi_5min"))


def build_model(checkpoint: dict, schema: DataSchema, physics_norm: dict, device: str) -> SerialPhysicsForecaster:
    model_config = checkpoint.get("model_config", {})
    model = SerialPhysicsForecaster(
        serial_input_dim=len(schema.serial_feature_columns),
        weather_input_dim=int(model_config.get("weather_input_dim", len(schema.weather_feature_columns))),
        physics_input_dim=len(schema.physics_feature_columns),
        serial_hidden=int(model_config.get("serial_hidden", 64)),
        weather_hidden=int(model_config.get("weather_hidden", 32)),
        physics_hidden=int(model_config.get("physics_hidden", 64)),
        fusion_hidden=int(model_config.get("fusion_hidden", 128)),
        dropout=float(model_config.get("dropout", 0.1)),
        use_rest2_calibration=bool(model_config.get("use_rest2_calibration", False)),
        use_clear_sky_weather_head=bool(model_config.get("use_clear_sky_weather_head", False)),
        use_weather_prior_fusion=bool(model_config.get("use_weather_prior_fusion", False)),
        use_clear_sky_power_prior=bool(model_config.get("use_clear_sky_power_prior", False)),
        weather_prior_weight_max=float(model_config.get("weather_prior_weight_max", 1.0)),
        sky_index_max=float(model_config.get("sky_index_max", 2.0)),
        target_column=model_config.get("target_column", schema.target_column),
        physics_feature_columns=tuple(schema.physics_feature_columns),
        physics_norm=model_config.get("physics_norm") or _serialize_norm(physics_norm),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def rest2_calibration_state(model: SerialPhysicsForecaster) -> dict[str, float]:
    if model.rest2_calibrator is None:
        return {
            "rest2_aod_scale": 1.0,
            "rest2_pwv_scale": 1.0,
            "rest2_pressure_scale": 1.0,
            "rest2_blend": 0.0,
        }
    calibrator = model.rest2_calibrator
    return {
        "rest2_aod_scale": float(torch.exp(calibrator.log_aod_scale).detach().cpu()),
        "rest2_pwv_scale": float(torch.exp(calibrator.log_pwv_scale).detach().cpu()),
        "rest2_pressure_scale": float(torch.exp(calibrator.log_pressure_scale).detach().cpu()),
        "rest2_blend": float(torch.sigmoid(calibrator.blend_logit).detach().cpu()),
    }


def lookup_diagnostics(csv_path: str | Path, schema: DataSchema, timestamps: list[str]) -> pd.DataFrame:
    frame = read_training_csv(csv_path, schema.timestamp_column)
    frame["_timestamp_key"] = frame[schema.timestamp_column].astype(str)
    lookup = frame.set_index("_timestamp_key", drop=False)
    rows = []
    for timestamp in timestamps:
        row = lookup.loc[timestamp]
        values = {
            schema.timestamp_column: timestamp,
            "target_column": schema.target_column,
        }
        for column in DIAGNOSTIC_COLUMNS:
            values[column] = row[column] if column in row.index else None
        rows.append(values)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    checkpoint = _load_checkpoint(args.checkpoint, args.device)
    schema = schema_from_checkpoint(checkpoint, args.target_column)
    serial_norm = _deserialize_norm(checkpoint.get("serial_norm"))
    physics_norm = _deserialize_norm(checkpoint.get("physics_norm"))
    _, _, test_loader, _, eval_physics_norm = build_loaders(
        args.csv,
        schema,
        args.window_steps,
        args.batch_size,
        serial_norm=serial_norm,
        physics_norm=physics_norm,
    )
    physics_norm_for_model = physics_norm or eval_physics_norm
    model = build_model(checkpoint, schema, physics_norm_for_model, args.device)
    calibration_state = rest2_calibration_state(model)

    records = []
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in test_loader:
            tensor_batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(tensor_batch)
            pred = outputs["prediction"].detach().cpu().view(-1)
            target = tensor_batch["target"].detach().cpu().view(-1)
            predictions.append(pred)
            targets.append(target)
            for idx, timestamp in enumerate(batch["timestamp"]):
                record = {
                    "dtime": timestamp,
                    "prediction": float(pred[idx]),
                    "target": float(target[idx]),
                    "absolute_error": float(abs(pred[idx] - target[idx])),
                    **calibration_state,
                }
                for name in (
                    "clear_sky_base",
                    "k_pred",
                    "residual_pred",
                    "serial_pred",
                    "prior_pred",
                    "weather_prior",
                    "k_weather",
                    "weather_k_prior",
                    "weather_adjusted_prior",
                    "prior_weight",
                    "prior_weight_raw",
                    "baseline_prediction",
                    "clear_sky_ghi",
                    "clear_sky_power_prior",
                    "power_scale",
                    "power_bias",
                    "rest2_gate",
                    "rest2_effective_blend",
                ):
                    if name in outputs:
                        record[name] = float(outputs[name].detach().cpu().view(-1)[idx])
                if "baseline_prediction" in record:
                    record["absolute_error_delta"] = record["absolute_error"] - abs(
                        record["baseline_prediction"] - record["target"]
                    )
                records.append(record)

    pred_tensor = torch.cat(predictions, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    full_frame = read_training_csv(args.csv, schema.timestamp_column)
    full_target = pd.to_numeric(full_frame[schema.target_column], errors="coerce")
    full_target = full_target[pd.notna(full_target)]
    y_max = float(full_target.max()) if len(full_target) else float("nan")
    metrics = regression_metrics(pred_tensor.view(-1, 1), target_tensor.view(-1, 1), y_max=y_max)
    diagnostics = lookup_diagnostics(args.csv, schema, [row["dtime"] for row in records])
    output = pd.DataFrame(records).merge(diagnostics, on="dtime", how="left")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"wrote={output_path} rows={len(output)}")
    print(metrics)


if __name__ == "__main__":
    main()
