from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_ablation import add_decisions, build_model_selection, write_model_selection_yaml, write_report
from diagnose_physics_consistency import make_work_frame, summarize_frame, summarize_segments
from rest2_physics_fusion.data.schema import DataSchema, validate_columns
from rest2_physics_fusion.training.model_selection import MODEL_VARIANTS, dataset_key, resolve_model_variant
from run_ablation import build_summary, discover_csvs, load_config, train_and_eval_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full model-ready CSV workflow: schema check, physics consistency diagnosis, "
            "baseline/physics ablation, decision analysis, and model-selection table export."
        )
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--csv-dir", default=str(ROOT / "data" / "mock_model_ready"))
    parser.add_argument("--csv-files", nargs="*", default=None, help="CSV file names under --csv-dir or absolute CSV paths.")
    parser.add_argument("--pattern", default="train_*.csv")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument(
        "--station-only",
        action="store_true",
        help="Only run station-level model_ready CSVs; auxiliary weather/forecast CSVs are ignored.",
    )
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "dataset_pipeline"))
    parser.add_argument("--target-columns", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "baseline",
            "physics_feature_only",
            "clear_sky_power_prior",
            "weather_prior_weak",
            "weather_prior",
            "rest2_calibrated",
            "weather_prior_rest2",
        ],
        choices=sorted(MODEL_VARIANTS),
    )
    parser.add_argument(
        "--selection-allowed-model-types",
        nargs="*",
        default=[
            "physics_feature_only",
            "clear_sky_power_prior",
            "weather_prior_weak",
            "rest2_calibrated",
            "weather_prior",
            "weather_prior_rest2",
        ],
        choices=sorted(MODEL_VARIANTS),
    )
    parser.add_argument("--selection-default-model-type", default="baseline", choices=sorted(MODEL_VARIANTS))
    parser.add_argument("--selection-include-watch", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-physics-diagnostics", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--train-selected", action="store_true")
    parser.add_argument("--day-clear-threshold", type=float, default=20.0)
    parser.add_argument("--upper-bound-tolerance", type=float, default=1.15)
    parser.add_argument("--night-target-threshold", type=float, default=5.0)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    cwd_candidate = Path.cwd() / value
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate
    return ROOT / value


def validate_model_ready_csvs(csv_files: list[Path], target_columns: list[str]) -> pd.DataFrame:
    rows = []
    for csv_path in csv_files:
        columns = set(pd.read_csv(csv_path, nrows=5).columns)
        for target_column in target_columns:
            schema = DataSchema(target_column=target_column)
            try:
                validate_columns(columns, schema)
                status = "ok"
                missing = ""
            except ValueError as exc:
                status = "missing_columns"
                missing = str(exc)
            rows.append(
                {
                    "dataset": csv_path.name,
                    "target_column": target_column,
                    "schema_status": status,
                    "missing_detail": missing,
                }
            )
    validation = pd.DataFrame(rows)
    failed = validation[validation["schema_status"] != "ok"]
    if not failed.empty:
        raise ValueError("Some CSV files are not model-ready:\n" + failed.to_string(index=False))
    return validation


def run_physics_diagnostics(
    csv_files: list[Path],
    target_columns: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    summary_rows = []
    segment_frames = []
    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        for target_column in target_columns:
            if target_column not in frame.columns:
                continue
            work = make_work_frame(frame, target_column, args.day_clear_threshold)
            summary_rows.append(
                summarize_frame(
                    work,
                    dataset=csv_path.name,
                    target_column=target_column,
                    upper_bound_tolerance=args.upper_bound_tolerance,
                    night_target_threshold=args.night_target_threshold,
                )
            )
            segment_frames.append(
                summarize_segments(
                    work,
                    dataset=csv_path.name,
                    target_column=target_column,
                    upper_bound_tolerance=args.upper_bound_tolerance,
                    night_target_threshold=args.night_target_threshold,
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / "physics_consistency_summary.csv", index=False, encoding="utf-8-sig")
    if segment_frames:
        pd.concat(segment_frames, ignore_index=True).to_csv(
            output_dir / "physics_consistency_segments.csv",
            index=False,
            encoding="utf-8-sig",
        )


def run_ablation_matrix(
    csv_files: list[Path],
    target_columns: list[str],
    output_root: Path,
    cfg: dict,
    seeds: list[int],
    variants: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    variant_defs = [(name, resolve_model_variant(name)) for name in variants]
    if "baseline" not in {name for name, _ in variant_defs}:
        variant_defs.insert(0, ("baseline", resolve_model_variant("baseline")))

    for csv_path in csv_files:
        key = dataset_key(csv_path)
        for target_column in target_columns:
            for seed in seeds:
                for model_type, variant in variant_defs:
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
    results_df.to_csv(output_root / "ablation_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_root / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    return results_df, summary_df


def run_decision_analysis(summary_df: pd.DataFrame, results_path: Path, output_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    analysis_args = SimpleNamespace(
        min_mae_gain=0.5,
        min_mae_win_rate=0.6,
        max_delta_std_ratio=1.5,
        selection_default_model_type=args.selection_default_model_type,
        selection_allowed_model_types=args.selection_allowed_model_types,
        selection_include_watch=args.selection_include_watch,
    )
    decisions = add_decisions(summary_df, analysis_args)
    output_root.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_root / "ablation_decisions.csv", index=False, encoding="utf-8-sig")
    decisions[decisions["decision"] == "promote_candidate"].to_csv(
        output_root / "promote_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decisions[decisions["decision"] == "watch_high_variance"].to_csv(
        output_root / "watch_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report_args = SimpleNamespace(**vars(analysis_args))
    write_report(decisions, output_root, report_args)
    selection = build_model_selection(decisions, report_args)
    selection.to_csv(output_root / "model_selection.csv", index=False, encoding="utf-8-sig")
    write_model_selection_yaml(selection, output_root / "model_selection.yaml")
    print(f"results_csv={results_path}")
    print(f"summary_csv={output_root / 'ablation_summary.csv'}")
    print(f"decisions_csv={output_root / 'ablation_decisions.csv'}")
    print(f"model_selection_csv={output_root / 'model_selection.csv'}")
    print(decisions[["dataset", "target_column", "comparison_model_type", "mae_gain", "rmse_gain", "candidate_win_rate_mae", "decision"]].to_string(index=False))
    return selection


def train_selected_models(
    selection: pd.DataFrame,
    csv_files: list[Path],
    output_root: Path,
    cfg: dict,
    seed: int,
) -> pd.DataFrame:
    csv_by_name = {path.name: path for path in csv_files}
    csv_by_key = {dataset_key(path): path for path in csv_files}
    rows = []
    for record in selection.to_dict(orient="records"):
        dataset = str(record["dataset"])
        csv_path = csv_by_name.get(dataset) or csv_by_key.get(dataset)
        if csv_path is None:
            available = sorted({*csv_by_name.keys(), *csv_by_key.keys()})
            raise FileNotFoundError(
                f"Cannot match selected dataset={dataset!r} to a CSV file. "
                f"Available datasets/keys: {available}"
            )
        model_type = str(record["selected_model_type"])
        variant = resolve_model_variant(model_type)
        target_column = str(record["target_column"])
        run_dir = output_root / dataset_key(csv_path) / target_column / model_type
        try:
            result = train_and_eval_one(
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
        except Exception as exc:
            raise RuntimeError(
                f"Failed to train selected model for dataset={dataset} "
                f"target_column={target_column} model_type={model_type} output_dir={run_dir}"
            ) from exc
        checkpoint = Path(str(result.get("checkpoint", "")))
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Selected model did not create best.pt: dataset={dataset} "
                f"target_column={target_column} model_type={model_type} expected={checkpoint}"
            )
        rows.append(result)
    trained = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    trained.to_csv(output_root / "selected_model_results.csv", index=False, encoding="utf-8-sig")
    return trained


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
    csv_files = discover_csvs(
        csv_dir,
        args.pattern,
        args.datasets,
        args.max_datasets,
        args.csv_files,
        station_only=args.station_only,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    validation = validate_model_ready_csvs(csv_files, target_columns)
    validation.to_csv(output_root / "schema_validation.csv", index=False, encoding="utf-8-sig")
    print(f"schema_validation_csv={output_root / 'schema_validation.csv'}")

    if not args.skip_physics_diagnostics:
        physics_dir = output_root / "physics_consistency"
        run_physics_diagnostics(csv_files, target_columns, physics_dir, args)
        print(f"physics_consistency_summary={physics_dir / 'physics_consistency_summary.csv'}")

    selection = pd.DataFrame()
    if not args.skip_ablation:
        results_df, summary_df = run_ablation_matrix(
            csv_files,
            target_columns,
            output_root,
            cfg,
            seeds,
            args.variants,
        )
        if not args.skip_analysis:
            selection = run_decision_analysis(summary_df, output_root / "ablation_results.csv", output_root, args)
        else:
            print(f"results_csv={output_root / 'ablation_results.csv'}")
            print(f"summary_csv={output_root / 'ablation_summary.csv'}")

    if args.train_selected:
        if selection.empty:
            selection_path = output_root / "model_selection.csv"
            if not selection_path.exists():
                raise FileNotFoundError("Cannot --train-selected without model_selection.csv. Run analysis first.")
            selection = pd.read_csv(selection_path)
        trained = train_selected_models(selection, csv_files, output_root / "selected_models", cfg, seeds[0])
        print(f"selected_model_results={output_root / 'selected_models' / 'selected_model_results.csv'}")
        print(trained.to_string(index=False))


if __name__ == "__main__":
    main()
