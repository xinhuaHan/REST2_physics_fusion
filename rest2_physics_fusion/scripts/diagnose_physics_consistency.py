from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_TO_HORIZON = {
    "target_ghi_5min": "5min",
    "target_ghi_4h": "4h",
    "target_ghi_1d": "1d",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default=str(root / "data" / "mock_model_ready"))
    parser.add_argument("--pattern", default="train_*.csv")
    parser.add_argument("--csv-files", nargs="*", default=None)
    parser.add_argument("--target-columns", nargs="*", default=list(TARGET_TO_HORIZON))
    parser.add_argument("--output-dir", default=str(root / "outputs" / "physics_consistency"))
    parser.add_argument("--day-clear-threshold", type=float, default=20.0)
    parser.add_argument("--upper-bound-tolerance", type=float, default=1.15)
    parser.add_argument("--night-target-threshold", type=float, default=5.0)
    return parser.parse_args()


def resolve_files(csv_dir: Path, pattern: str, csv_files: list[str] | None) -> list[Path]:
    if csv_files:
        files = [Path(path) for path in csv_files]
        return [path if path.is_absolute() else csv_dir / path for path in files]
    return sorted(csv_dir.glob(pattern))


def horizon_columns(target_column: str) -> tuple[str, str, str]:
    suffix = TARGET_TO_HORIZON.get(target_column)
    if suffix is None:
        return "ghi_clear_target", "weather_adjusted_clear_sky_ghi", "mu0_target"
    return (
        f"ghi_clear_horizon_{suffix}",
        f"weather_adjusted_clear_sky_ghi_horizon_{suffix}",
        f"mu0_horizon_{suffix}",
    )


def safe_numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 3:
        return float("nan")
    if valid.iloc[:, 0].std() <= 1e-9 or valid.iloc[:, 1].std() <= 1e-9:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


def regression_metrics(pred: pd.Series, target: pd.Series) -> tuple[float, float, float]:
    error = (pred - target).dropna()
    if len(error) == 0:
        return float("nan"), float("nan"), float("nan")
    mae = float(error.abs().mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    bias = float(error.mean())
    return mae, rmse, bias


def make_work_frame(frame: pd.DataFrame, target_column: str, day_clear_threshold: float) -> pd.DataFrame:
    clear_col, weather_prior_col, mu0_col = horizon_columns(target_column)
    out = pd.DataFrame(index=frame.index)
    out["target"] = safe_numeric(frame, target_column)
    out["clear_sky"] = safe_numeric(frame, clear_col).clip(lower=0.0)
    out["weather_prior"] = safe_numeric(frame, weather_prior_col).clip(lower=0.0)
    out["mu0"] = safe_numeric(frame, mu0_col).clip(lower=0.0)
    out["current_ghi"] = safe_numeric(frame, "input_ghi").clip(lower=0.0)
    out["weather_attenuation_prior"] = safe_numeric(frame, "weather_attenuation_prior")
    out["clear_sky_index_current"] = safe_numeric(frame, "clear_sky_index_current")
    out["weather_is_joined"] = safe_numeric(frame, "weather_is_joined", default=0.0)
    out["target_clear_sky_index"] = out["target"] / out["clear_sky"].clip(lower=1.0)
    out["target_weather_prior_index"] = out["target"] / out["weather_prior"].clip(lower=1.0)
    out["is_day"] = out["clear_sky"] > day_clear_threshold
    out["day_segment"] = np.where(out["is_day"], "day", "night_or_low_sun")
    out["clear_sky_bin"] = pd.cut(
        out["clear_sky"],
        bins=[-0.1, day_clear_threshold, 200.0, 500.0, float("inf")],
        labels=["low_or_night", "weak_sun", "medium_sun", "strong_sun"],
    ).astype(str)
    out["sky_index_bin"] = pd.cut(
        out["clear_sky_index_current"].fillna(0.0),
        bins=[-0.1, 0.2, 0.8, 1.2, float("inf")],
        labels=["very_cloudy_or_night", "cloudy", "near_clear", "over_clear_or_noisy"],
    ).astype(str)
    return out


def summarize_frame(
    work: pd.DataFrame,
    *,
    dataset: str,
    target_column: str,
    upper_bound_tolerance: float,
    night_target_threshold: float,
) -> dict[str, object]:
    day = work[work["is_day"]]
    night = work[~work["is_day"]]
    clear_mae, clear_rmse, clear_bias = regression_metrics(work["clear_sky"], work["target"])
    weather_mae, weather_rmse, weather_bias = regression_metrics(work["weather_prior"], work["target"])
    persistence_mae, persistence_rmse, persistence_bias = regression_metrics(work["current_ghi"], work["target"])
    target = work["target"]
    clear_sky = work["clear_sky"]
    weather_prior = work["weather_prior"]
    upper_exceed = target > clear_sky * upper_bound_tolerance
    weather_upper_exceed = target > weather_prior * upper_bound_tolerance
    night_nonzero = night["target"] > night_target_threshold
    day_scale = day["target"] / day["clear_sky"].clip(lower=1.0)
    target_range = float(target.quantile(0.99) - target.quantile(0.01)) if len(target.dropna()) else float("nan")
    row = {
        "dataset": dataset,
        "target_column": target_column,
        "rows": int(len(work)),
        "day_rows": int(len(day)),
        "night_rows": int(len(night)),
        "day_fraction": float(len(day) / max(1, len(work))),
        "target_mean": float(target.mean()),
        "clear_sky_mean": float(clear_sky.mean()),
        "weather_prior_mean": float(weather_prior.mean()),
        "target_clear_sky_corr": safe_corr(target, clear_sky),
        "target_weather_prior_corr": safe_corr(target, weather_prior),
        "target_current_ghi_corr": safe_corr(target, work["current_ghi"]),
        "clear_sky_mae": clear_mae,
        "clear_sky_rmse": clear_rmse,
        "clear_sky_bias": clear_bias,
        "weather_prior_mae": weather_mae,
        "weather_prior_rmse": weather_rmse,
        "weather_prior_bias": weather_bias,
        "current_ghi_mae": persistence_mae,
        "current_ghi_rmse": persistence_rmse,
        "current_ghi_bias": persistence_bias,
        "weather_prior_mae_gain_vs_clear_sky": clear_mae - weather_mae,
        "weather_prior_mae_gain_vs_current_ghi": persistence_mae - weather_mae,
        "clear_sky_upper_exceed_rate": float(upper_exceed.mean()),
        "weather_prior_upper_exceed_rate": float(weather_upper_exceed.mean()),
        "night_nonzero_target_rate": float(night_nonzero.mean()) if len(night) else float("nan"),
        "night_target_mean": float(night["target"].mean()) if len(night) else float("nan"),
        "target_clear_sky_index_mean_day": float(day["target_clear_sky_index"].mean()) if len(day) else float("nan"),
        "target_clear_sky_scale_median_day": float(day_scale.median()) if len(day) else float("nan"),
        "target_clear_sky_scale_iqr_day": float(day_scale.quantile(0.75) - day_scale.quantile(0.25)) if len(day) else float("nan"),
        "target_clear_sky_index_p95_day": float(day["target_clear_sky_index"].quantile(0.95)) if len(day) else float("nan"),
        "target_clear_sky_index_gt_1p2_day_rate": float((day["target_clear_sky_index"] > 1.2).mean()) if len(day) else float("nan"),
        "target_p99": float(target.quantile(0.99)),
        "target_dynamic_range_p01_p99": target_range,
        "power_like_score": power_like_score(
            target_current_corr=safe_corr(target, work["current_ghi"]),
            scale_median=float(day_scale.median()) if len(day) else float("nan"),
            scale_iqr=float(day_scale.quantile(0.75) - day_scale.quantile(0.25)) if len(day) else float("nan"),
            upper_exceed_rate=float(upper_exceed.mean()),
            night_nonzero_rate=float(night_nonzero.mean()) if len(night) else float("nan"),
        ),
        "weather_joined_rate": float(work["weather_is_joined"].fillna(0.0).mean()),
    }
    row.update(classify_physical_consistency(row))
    return row


def power_like_score(
    *,
    target_current_corr: float,
    scale_median: float,
    scale_iqr: float,
    upper_exceed_rate: float,
    night_nonzero_rate: float,
) -> float:
    score = 0.0
    if np.isfinite(target_current_corr) and target_current_corr >= 0.6:
        score += 0.3
    if np.isfinite(scale_median) and (scale_median < 0.5 or scale_median > 1.5):
        score += 0.25
    if np.isfinite(scale_iqr) and scale_iqr <= 0.75:
        score += 0.15
    if np.isfinite(upper_exceed_rate) and upper_exceed_rate > 0.1:
        score += 0.15
    if np.isfinite(night_nonzero_rate) and night_nonzero_rate <= 0.1:
        score += 0.15
    return float(min(score, 1.0))


def classify_physical_consistency(row: dict[str, object]) -> dict[str, object]:
    upper_rate = float(row.get("clear_sky_upper_exceed_rate", 0.0))
    night_rate = float(row.get("night_nonzero_target_rate", 0.0))
    weather_corr = float(row.get("target_weather_prior_corr", np.nan))
    weather_gain_vs_current = float(row.get("weather_prior_mae_gain_vs_current_ghi", np.nan))
    power_score = float(row.get("power_like_score", 0.0))
    warnings = []
    if upper_rate > 0.2:
        warnings.append("many_targets_above_clear_sky")
    if night_rate > 0.2:
        warnings.append("many_nonzero_night_targets")
    if np.isfinite(weather_corr) and weather_corr < 0.6:
        warnings.append("weak_weather_prior_correlation")
    if np.isfinite(weather_gain_vs_current) and weather_gain_vs_current < 0.0:
        warnings.append("weather_prior_worse_than_current_ghi")
    if power_score >= 0.6:
        warnings.append("target_may_need_clear_sky_power_prior")

    if upper_rate > 0.2 or night_rate > 0.2:
        validity = "poor_for_rest2_validation"
    elif np.isfinite(weather_corr) and weather_corr >= 0.8:
        validity = "reasonable_for_weather_prior_validation"
    else:
        validity = "needs_real_data_or_more_features"
    return {
        "physics_consistency_warning": "|".join(warnings) if warnings else "ok",
        "rest2_validation_validity": validity,
        "power_prior_recommendation": "try_clear_sky_power_prior" if power_score >= 0.6 else "not_enough_evidence",
    }


def summarize_segments(
    work: pd.DataFrame,
    *,
    dataset: str,
    target_column: str,
    upper_bound_tolerance: float,
    night_target_threshold: float,
) -> pd.DataFrame:
    rows = []
    for segment_type, column in [
        ("day_segment", "day_segment"),
        ("clear_sky_bin", "clear_sky_bin"),
        ("sky_index_bin", "sky_index_bin"),
    ]:
        for segment_value, group in work.groupby(column, dropna=False):
            row = summarize_frame(
                group,
                dataset=dataset,
                target_column=target_column,
                upper_bound_tolerance=upper_bound_tolerance,
                night_target_threshold=night_target_threshold,
            )
            row["segment_type"] = segment_type
            row["segment_value"] = str(segment_value)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    files = resolve_files(csv_dir, args.pattern, args.csv_files)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir} with pattern={args.pattern!r}")

    summary_rows = []
    segment_frames = []
    for path in files:
        frame = pd.read_csv(path)
        for target_column in args.target_columns:
            if target_column not in frame.columns:
                continue
            work = make_work_frame(frame, target_column, args.day_clear_threshold)
            summary_rows.append(
                summarize_frame(
                    work,
                    dataset=path.name,
                    target_column=target_column,
                    upper_bound_tolerance=args.upper_bound_tolerance,
                    night_target_threshold=args.night_target_threshold,
                )
            )
            segment_frames.append(
                summarize_segments(
                    work,
                    dataset=path.name,
                    target_column=target_column,
                    upper_bound_tolerance=args.upper_bound_tolerance,
                    night_target_threshold=args.night_target_threshold,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    segments = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    summary_path = output_dir / "physics_consistency_summary.csv"
    segment_path = output_dir / "physics_consistency_segments.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    segments.to_csv(segment_path, index=False, encoding="utf-8-sig")
    print(f"summary_csv={summary_path}")
    print(f"segments_csv={segment_path}")
    display_columns = [
        "dataset",
        "target_column",
        "target_weather_prior_corr",
        "weather_prior_mae",
        "current_ghi_mae",
        "weather_prior_mae_gain_vs_current_ghi",
        "clear_sky_upper_exceed_rate",
        "night_nonzero_target_rate",
        "target_clear_sky_scale_median_day",
        "power_like_score",
        "power_prior_recommendation",
        "rest2_validation_validity",
        "physics_consistency_warning",
    ]
    print(summary[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
