from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.training.train import train_model
from rest2_physics_fusion.data.schema import DataSchema
from rest2_physics_fusion.training.model_selection import MODEL_VARIANTS, resolve_model_variant, select_model_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--train-csv", default=str(ROOT / "data" / "mock_model_ready" / "train_jiefang_station.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--model-type", default=None, choices=sorted(MODEL_VARIANTS))
    parser.add_argument("--model-selection-csv", default=None)
    parser.add_argument("--use-rest2-calibration", action="store_true")
    parser.add_argument("--use-weather-prior-fusion", action="store_true")
    parser.add_argument("--prior-weight-l1", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg["training"]
    target_column = args.target_column or data_cfg.get("target_column", "target_ghi_5min")
    model_type = args.model_type
    if model_type is None and args.model_selection_csv:
        model_type = select_model_type(
            args.model_selection_csv,
            csv_path=args.train_csv,
            target_column=target_column,
            default_model_type="baseline",
        )
    variant = resolve_model_variant(model_type or "baseline")
    schema = DataSchema(target_column=target_column)
    output_dir = Path(args.output_dir or train_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    best = train_model(
        args.train_csv,
        output_dir=output_dir,
        schema=schema,
        window_steps=int(data_cfg["window_steps"]),
        batch_size=int(train_cfg["batch_size"]),
        epochs=int(train_cfg["epochs"]),
        lr=float(train_cfg["lr"]),
        device=str(train_cfg["device"]),
        serial_hidden=int(model_cfg.get("serial_hidden", 64)),
        weather_hidden=int(model_cfg.get("weather_hidden", 32)),
        physics_hidden=int(model_cfg.get("physics_hidden", 64)),
        fusion_hidden=int(model_cfg.get("fusion_hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_rest2_calibration=bool(
            variant.use_rest2_calibration
            or args.use_rest2_calibration
            or model_cfg.get("use_rest2_calibration", False)
        ),
        use_clear_sky_weather_head=bool(
            variant.use_clear_sky_weather_head
            and str(model_cfg.get("architecture", "")).lower() == "clear_sky_weather_fusion"
        ),
        use_weather_prior_fusion=bool(
            variant.use_weather_prior_fusion
            or args.use_weather_prior_fusion
            or model_cfg.get("use_weather_prior_fusion", False)
        ),
        use_clear_sky_power_prior=bool(variant.use_clear_sky_power_prior),
        weather_prior_weight_max=float(variant.weather_prior_weight_max),
        prior_weight_l1=float(
            args.prior_weight_l1
            if args.prior_weight_l1 is not None
            else model_cfg.get("prior_weight_l1", 0.0)
        ),
        sky_index_max=float(cfg.get("physics", {}).get("sky_index_max", 2.0)),
    )
    print(f"model_type={model_type or 'baseline'}")
    print(f"best_checkpoint={best}")


if __name__ == "__main__":
    main()
