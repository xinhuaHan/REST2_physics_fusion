from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--rest2-csv", default=None, help="Backward-compatible candidate CSV argument.")
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--day-clear-threshold", type=float, default=20.0)
    return parser.parse_args()


def read_diag(path: str | Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dtime", "prediction", "target", "absolute_error", "clear_sky_base"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing diagnostic columns: {missing}")
    keep_as_shared = {
        "dtime",
        "target",
        "target_column",
        "source_type",
        "station_name",
        "input_ghi",
        "clear_sky_base",
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
        "weather_adjusted_clear_sky_ghi_horizon_5min",
        "weather_adjusted_clear_sky_ghi_horizon_4h",
        "weather_adjusted_clear_sky_ghi_horizon_1d",
    }
    rename = {
        column: f"{column}_{suffix}"
        for column in df.columns
        if column not in keep_as_shared
    }
    return df.rename(columns=rename)


def compare_rows(baseline: pd.DataFrame, rest2: pd.DataFrame) -> pd.DataFrame:
    merged = baseline.merge(rest2, on="dtime", how="inner", suffixes=("", "_rest2_shared"))
    if len(merged) == 0:
        raise ValueError("No matching dtime rows between baseline and REST2 diagnostics")
    default_columns = (
        "k_pred_baseline",
        "k_pred_rest2",
        "residual_pred_baseline",
        "residual_pred_rest2",
        "rest2_gate_rest2",
        "rest2_effective_blend_rest2",
        "serial_pred_baseline",
        "serial_pred_rest2",
        "prior_pred_baseline",
        "prior_pred_rest2",
        "k_weather_baseline",
        "k_weather_rest2",
        "weather_k_prior_baseline",
        "weather_k_prior_rest2",
        "prior_weight_baseline",
        "prior_weight_rest2",
    )
    for column in default_columns:
        if column not in merged.columns:
            merged[column] = 0.0
    merged["error_delta_rest2_minus_baseline"] = merged["absolute_error_rest2"] - merged["absolute_error_baseline"]
    merged["rest2_improved"] = merged["error_delta_rest2_minus_baseline"] < 0.0
    merged["prediction_delta_rest2_minus_baseline"] = merged["prediction_rest2"] - merged["prediction_baseline"]
    merged["k_delta_rest2_minus_baseline"] = merged.get("k_pred_rest2", 0.0) - merged.get("k_pred_baseline", 0.0)
    merged["residual_delta_rest2_minus_baseline"] = merged.get("residual_pred_rest2", 0.0) - merged.get("residual_pred_baseline", 0.0)
    return merged


def segment_frame(df: pd.DataFrame, day_clear_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["day_segment"] = out["clear_sky_base"].where(out["clear_sky_base"] > day_clear_threshold, 0.0)
    out["day_segment"] = out["day_segment"].apply(lambda value: "day" if value > day_clear_threshold else "night_or_low_sun")
    out["clear_sky_bin"] = pd.cut(
        out["clear_sky_base"],
        bins=[-0.1, day_clear_threshold, 200.0, 500.0, float("inf")],
        labels=["low_or_night", "weak_sun", "medium_sun", "strong_sun"],
    ).astype(str)
    out["sky_index_bin"] = pd.cut(
        out["clear_sky_index_current"].fillna(0.0),
        bins=[-0.1, 0.2, 0.8, 1.2, float("inf")],
        labels=["very_cloudy_or_night", "cloudy", "near_clear", "over_clear_or_noisy"],
    ).astype(str)
    out["prior_weight_bin"] = pd.cut(
        out["prior_weight_rest2"].fillna(0.0),
        bins=[-0.1, 0.05, 0.2, 0.5, 1.0],
        labels=["none", "low", "medium", "high"],
    ).astype(str)
    return out


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        row = {
            "rows": len(df),
            "rest2_improved_rate": float(df["rest2_improved"].mean()),
            "baseline_mae": float(df["absolute_error_baseline"].mean()),
            "rest2_mae": float(df["absolute_error_rest2"].mean()),
            "mae_delta_rest2_minus_baseline": float(df["error_delta_rest2_minus_baseline"].mean()),
            "mae_delta_std": float(df["error_delta_rest2_minus_baseline"].std()),
            "prediction_delta_mean": float(df["prediction_delta_rest2_minus_baseline"].mean()),
            "k_baseline_mean": float(df["k_pred_baseline"].mean()),
            "k_rest2_mean": float(df["k_pred_rest2"].mean()),
            "residual_baseline_mean": float(df["residual_pred_baseline"].mean()),
            "residual_rest2_mean": float(df["residual_pred_rest2"].mean()),
            "rest2_gate_mean": float(df["rest2_gate_rest2"].mean()),
            "rest2_effective_blend_mean": float(df["rest2_effective_blend_rest2"].mean()),
            "serial_pred_baseline_mean": float(df["serial_pred_baseline"].mean()),
            "serial_pred_rest2_mean": float(df["serial_pred_rest2"].mean()),
            "prior_pred_baseline_mean": float(df["prior_pred_baseline"].mean()),
            "prior_pred_rest2_mean": float(df["prior_pred_rest2"].mean()),
            "k_weather_baseline_mean": float(df["k_weather_baseline"].mean()),
            "k_weather_rest2_mean": float(df["k_weather_rest2"].mean()),
            "weather_k_prior_baseline_mean": float(df["weather_k_prior_baseline"].mean()),
            "weather_k_prior_rest2_mean": float(df["weather_k_prior_rest2"].mean()),
            "prior_weight_baseline_mean": float(df["prior_weight_baseline"].mean()),
            "prior_weight_rest2_mean": float(df["prior_weight_rest2"].mean()),
        }
        row["mae_gain"] = -row["mae_delta_rest2_minus_baseline"]
        return pd.DataFrame([row])
    grouped = df.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        rows=("dtime", "count"),
        rest2_improved_rate=("rest2_improved", "mean"),
        baseline_mae=("absolute_error_baseline", "mean"),
        rest2_mae=("absolute_error_rest2", "mean"),
        mae_delta_rest2_minus_baseline=("error_delta_rest2_minus_baseline", "mean"),
        mae_delta_std=("error_delta_rest2_minus_baseline", "std"),
        prediction_delta_mean=("prediction_delta_rest2_minus_baseline", "mean"),
        k_baseline_mean=("k_pred_baseline", "mean"),
        k_rest2_mean=("k_pred_rest2", "mean"),
        residual_baseline_mean=("residual_pred_baseline", "mean"),
        residual_rest2_mean=("residual_pred_rest2", "mean"),
        rest2_gate_mean=("rest2_gate_rest2", "mean"),
        rest2_effective_blend_mean=("rest2_effective_blend_rest2", "mean"),
        serial_pred_baseline_mean=("serial_pred_baseline", "mean"),
        serial_pred_rest2_mean=("serial_pred_rest2", "mean"),
        prior_pred_baseline_mean=("prior_pred_baseline", "mean"),
        prior_pred_rest2_mean=("prior_pred_rest2", "mean"),
        k_weather_baseline_mean=("k_weather_baseline", "mean"),
        k_weather_rest2_mean=("k_weather_rest2", "mean"),
        weather_k_prior_baseline_mean=("weather_k_prior_baseline", "mean"),
        weather_k_prior_rest2_mean=("weather_k_prior_rest2", "mean"),
        prior_weight_baseline_mean=("prior_weight_baseline", "mean"),
        prior_weight_rest2_mean=("prior_weight_rest2", "mean"),
    ).reset_index()
    summary["mae_gain"] = -summary["mae_delta_rest2_minus_baseline"]
    return summary


def calibration_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rest2_aod_scale_rest2",
        "rest2_pwv_scale_rest2",
        "rest2_pressure_scale_rest2",
        "rest2_blend_rest2",
        "rest2_gate_rest2",
        "rest2_effective_blend_rest2",
    ]
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        rows.append(
            {
                "parameter": column.replace("_rest2", ""),
                "mean": float(df[column].mean()),
                "min": float(df[column].min()),
                "max": float(df[column].max()),
            }
        )
    return pd.DataFrame(rows)


def add_candidate_summary_columns(summary: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    out = summary.copy()
    out["candidate_name"] = candidate_name
    aliases = {
        "candidate_improved_rate": "rest2_improved_rate",
        "candidate_mae": "rest2_mae",
        "candidate_gate_mean": "rest2_gate_mean",
        "candidate_effective_blend_mean": "rest2_effective_blend_mean",
        "candidate_k_mean": "k_rest2_mean",
        "candidate_residual_mean": "residual_rest2_mean",
        "serial_pred_candidate_mean": "serial_pred_rest2_mean",
        "prior_pred_candidate_mean": "prior_pred_rest2_mean",
        "k_weather_candidate_mean": "k_weather_rest2_mean",
        "weather_k_prior_candidate_mean": "weather_k_prior_rest2_mean",
        "prior_weight_candidate_mean": "prior_weight_rest2_mean",
    }
    for target, source in aliases.items():
        if source in out.columns:
            out[target] = out[source]
    return out


def main() -> None:
    args = parse_args()
    candidate_csv = args.candidate_csv or args.rest2_csv
    if not candidate_csv:
        raise ValueError("Provide --candidate-csv or the backward-compatible --rest2-csv")
    baseline = read_diag(args.baseline_csv, "baseline")
    rest2 = read_diag(candidate_csv, "rest2")
    compared = segment_frame(compare_rows(baseline, rest2), args.day_clear_threshold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_path = output_dir / "diagnostic_compare.csv"
    overall_path = output_dir / "diagnostic_summary_overall.csv"
    segment_path = output_dir / "diagnostic_summary_segments.csv"
    calibration_path = output_dir / "diagnostic_rest2_calibration.csv"

    compared.to_csv(row_path, index=False, encoding="utf-8-sig")
    add_candidate_summary_columns(summarize_group(compared, []), args.candidate_name).to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )
    segment_summary = pd.concat(
        [
            summarize_group(compared, ["day_segment"]).assign(segment_type="day_segment"),
            summarize_group(compared, ["clear_sky_bin"]).assign(segment_type="clear_sky_bin"),
            summarize_group(compared, ["sky_index_bin"]).assign(segment_type="sky_index_bin"),
            summarize_group(compared, ["prior_weight_bin"]).assign(segment_type="prior_weight_bin"),
        ],
        ignore_index=True,
    )
    add_candidate_summary_columns(segment_summary, args.candidate_name).to_csv(segment_path, index=False, encoding="utf-8-sig")
    calibration_summary(compared).to_csv(calibration_path, index=False, encoding="utf-8-sig")

    print(f"compare_csv={row_path}")
    print(f"overall_summary_csv={overall_path}")
    print(f"segment_summary_csv={segment_path}")
    print(f"calibration_csv={calibration_path}")
    print(pd.read_csv(overall_path).to_string(index=False))


if __name__ == "__main__":
    main()
