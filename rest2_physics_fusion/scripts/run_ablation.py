from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.data.preprocess import set_seed
from rest2_physics_fusion.data.schema import DataSchema
from rest2_physics_fusion.training.train import evaluate_model, train_model
from rest2_physics_fusion.training.model_selection import MODEL_VARIANTS, dataset_key, resolve_model_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--csv-dir", default=str(ROOT / "data" / "mock_model_ready"))
    parser.add_argument("--csv-files", nargs="*", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "ablation"))
    parser.add_argument("--pattern", default="train_*.csv")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-columns", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        choices=sorted(MODEL_VARIANTS),
        help="Model variants to compare. Default: all registered variants.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    cwd_candidate = Path.cwd() / value
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate
    return ROOT / value


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def discover_csvs(
    csv_dir: Path,
    pattern: str,
    datasets: list[str] | None,
    max_datasets: int | None,
    csv_files: list[str] | None = None,
) -> list[Path]:
    if csv_files:
        files = []
        for item in csv_files:
            path = Path(item)
            if not path.is_absolute():
                path = csv_dir / path
            files.append(path)
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing CSV files: {missing}")
    else:
        files = sorted(csv_dir.glob(pattern))
    if datasets:
        wanted = {name if name.endswith(".csv") else f"{name}.csv" for name in datasets}
        files = [path for path in files if path.name in wanted]
    if max_datasets is not None:
        files = files[: max(0, max_datasets)]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir} with pattern={pattern!r}")
    return files


def train_and_eval_one(
    *,
    csv_path: Path,
    output_dir: Path,
    model_type: str,
    use_rest2_calibration: bool,
    use_weather_prior_fusion: bool,
    use_clear_sky_weather_head: bool,
    use_clear_sky_power_prior: bool,
    weather_prior_weight_max: float,
    target_column: str,
    cfg: dict,
    seed: int,
) -> dict[str, object]:
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg["training"]
    schema = DataSchema(target_column=target_column)
    set_seed(seed)
    best_checkpoint = train_model(
        csv_path,
        output_dir=output_dir,
        schema=schema,
        window_steps=int(data_cfg["window_steps"]),
        batch_size=int(train_cfg["batch_size"]),
        epochs=int(train_cfg.get("epochs", 3)),
        lr=float(train_cfg["lr"]),
        device=str(train_cfg["device"]),
        serial_hidden=int(model_cfg.get("serial_hidden", 64)),
        weather_hidden=int(model_cfg.get("weather_hidden", 32)),
        physics_hidden=int(model_cfg.get("physics_hidden", 64)),
        fusion_hidden=int(model_cfg.get("fusion_hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_rest2_calibration=use_rest2_calibration,
        use_clear_sky_weather_head=bool(
            use_clear_sky_weather_head and str(model_cfg.get("architecture", "")).lower() == "clear_sky_weather_fusion"
        ),
        use_weather_prior_fusion=use_weather_prior_fusion,
        use_clear_sky_power_prior=use_clear_sky_power_prior,
        weather_prior_weight_max=weather_prior_weight_max,
        prior_weight_l1=float(model_cfg.get("prior_weight_l1", 0.0)),
        sky_index_max=float(cfg.get("physics", {}).get("sky_index_max", 2.0)),
    )
    metrics = evaluate_model(
        best_checkpoint,
        csv_path,
        schema=schema,
        window_steps=int(data_cfg["window_steps"]),
        batch_size=int(train_cfg["batch_size"]),
        device=str(train_cfg["device"]),
    )
    return {
        "dataset": csv_path.name,
        "dataset_key": dataset_key(csv_path),
        "target_column": target_column,
        "seed": seed,
        "model_type": model_type,
        "use_rest2_calibration": use_rest2_calibration,
        "use_weather_prior_fusion": use_weather_prior_fusion,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "checkpoint": str(best_checkpoint),
    }


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, target_column), group in results.groupby(["dataset", "target_column"]):
        pivot = group.pivot_table(index="seed", columns="model_type", values=["mae", "rmse"], aggfunc="first")
        if ("mae", "baseline") not in pivot.columns:
            continue
        baseline_mae = pivot[("mae", "baseline")]
        baseline_rmse = pivot[("rmse", "baseline")]
        model_types = sorted({column[1] for column in pivot.columns if column[0] == "mae"})
        for candidate in model_types:
            if candidate == "baseline" or ("mae", candidate) not in pivot.columns:
                continue
            candidate_mae = pivot[("mae", candidate)]
            candidate_rmse = pivot[("rmse", candidate)]
            delta_mae = candidate_mae - baseline_mae
            delta_rmse = candidate_rmse - baseline_rmse
            row = {
                "dataset": dataset,
                "target_column": target_column,
                "comparison_model_type": candidate,
                "num_seeds": int(len(delta_mae.dropna())),
                "baseline_mae_mean": float(baseline_mae.mean()),
                "baseline_mae_std": float(baseline_mae.std(ddof=0)),
                "candidate_mae_mean": float(candidate_mae.mean()),
                "candidate_mae_std": float(candidate_mae.std(ddof=0)),
                "delta_mae_mean": float(delta_mae.mean()),
                "delta_mae_std": float(delta_mae.std(ddof=0)),
                "candidate_win_rate_mae": float((delta_mae < 0.0).mean()),
                "baseline_rmse_mean": float(baseline_rmse.mean()),
                "baseline_rmse_std": float(baseline_rmse.std(ddof=0)),
                "candidate_rmse_mean": float(candidate_rmse.mean()),
                "candidate_rmse_std": float(candidate_rmse.std(ddof=0)),
                "delta_rmse_mean": float(delta_rmse.mean()),
                "delta_rmse_std": float(delta_rmse.std(ddof=0)),
                "candidate_win_rate_rmse": float((delta_rmse < 0.0).mean()),
            }
            if candidate == "rest2_calibrated":
                row.update(
                    {
                        "rest2_mae_mean": row["candidate_mae_mean"],
                        "rest2_mae_std": row["candidate_mae_std"],
                        "rest2_win_rate_mae": row["candidate_win_rate_mae"],
                        "rest2_rmse_mean": row["candidate_rmse_mean"],
                        "rest2_rmse_std": row["candidate_rmse_std"],
                        "rest2_win_rate_rmse": row["candidate_win_rate_rmse"],
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    csv_dir = resolve_path(args.csv_dir)
    output_root = resolve_path(args.output_root)
    cfg = load_config(config_path)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.device is not None:
        cfg["training"]["device"] = args.device
    target_columns = args.target_columns or [cfg.get("data", {}).get("target_column", "target_ghi_5min")]
    seeds = args.seeds or [int(cfg.get("project", {}).get("seed", 42))]

    csv_files = discover_csvs(csv_dir, args.pattern, args.datasets, args.max_datasets, args.csv_files)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    variant_names = args.variants or [
        "baseline",
        "physics_feature_only",
        "clear_sky_power_prior",
        "weather_prior_weak",
        "weather_prior",
        "rest2_calibrated",
        "weather_prior_rest2",
    ]
    if "baseline" not in variant_names:
        variant_names = ["baseline", *variant_names]
    variants = [(name, resolve_model_variant(name)) for name in variant_names]

    for csv_path in csv_files:
        key = dataset_key(csv_path)
        for target_column in target_columns:
            for seed in seeds:
                for model_type, variant in variants:
                    run_dir = output_root / key / target_column / f"seed_{seed}" / model_type
                    print(
                        f"dataset={csv_path.name} target_column={target_column} "
                        f"seed={seed} model_type={model_type} output_dir={run_dir}"
                    )
                    results.append(
                        train_and_eval_one(
                            csv_path=csv_path,
                            output_dir=run_dir,
                            model_type=model_type,
                            use_rest2_calibration=variant.use_rest2_calibration,
                            use_weather_prior_fusion=variant.use_weather_prior_fusion,
                            use_clear_sky_weather_head=variant.use_clear_sky_weather_head,
                            use_clear_sky_power_prior=variant.use_clear_sky_power_prior,
                            weather_prior_weight_max=variant.weather_prior_weight_max,
                            target_column=target_column,
                            cfg=cfg,
                            seed=seed,
                        )
                )

    results_df = pd.DataFrame(results)
    summary_df = build_summary(results_df)
    results_path = output_root / "ablation_results.csv"
    summary_path = output_root / "ablation_summary.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"results_csv={results_path}")
    print(f"summary_csv={summary_path}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
