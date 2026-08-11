"""Read-only preflight inspection for the production YLJ DNI/DHI Parquet."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_PARQUET = Path(
    "/data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/"
    "YLJ-Unified_format-with_DNI_DHI.parquet"
)
EXPECTED_COLUMNS = (
    "timestamp",
    "observe_power",
    "GHI_observe",
    "DNI_observe",
    "DHI_observe",
    "TEMP_observe",
    "WS_observe",
    "WD_observe",
    "PREC_observe",
    "PWAT_observe",
    "SDWE_observe",
    "GHI_forecast_1day",
    "TEMP_forecast_1day",
    "WS_forecast_1day",
    "WD_forecast_1day",
    "PREC_forecast_1day",
    "PWAT_forecast_1day",
    "SDWE_forecast_1day",
    "GHI_forecast_4hour",
    "TEMP_forecast_4hour",
    "WS_forecast_4hour",
    "WD_forecast_4hour",
    "PREC_forecast_4hour",
    "PWAT_forecast_4hour",
    "SDWE_forecast_4hour",
)
PWAT_COLUMNS = ("PWAT_observe", "PWAT_forecast_1day", "PWAT_forecast_4hour")
IRRADIANCE_COLUMNS = ("GHI_observe", "DNI_observe", "DHI_observe")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect YLJ schema, 15-minute cadence, DNI/DHI coverage and PWAT units "
            "without loading a model or changing the Parquet file."
        )
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--expected-step-minutes", type=int, default=15)
    parser.add_argument("--pwat-unit", choices=("mm", "cm"), default="mm")
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def numeric_summary(source: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(source, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = values[np.isfinite(values)]
    quantiles = (0.01, 0.05, 0.50, 0.95, 0.99)
    return {
        "rows": int(len(source)),
        "finite": int(len(finite)),
        "missing": int(source.isna().sum()),
        "parse_failures": int(numeric.isna().sum() - source.isna().sum()),
        "minimum": float(np.min(finite)) if len(finite) else None,
        "q01": float(np.quantile(finite, quantiles[0])) if len(finite) else None,
        "q05": float(np.quantile(finite, quantiles[1])) if len(finite) else None,
        "median": float(np.quantile(finite, quantiles[2])) if len(finite) else None,
        "mean": float(np.mean(finite)) if len(finite) else None,
        "q95": float(np.quantile(finite, quantiles[3])) if len(finite) else None,
        "q99": float(np.quantile(finite, quantiles[4])) if len(finite) else None,
        "maximum": float(np.max(finite)) if len(finite) else None,
        "negative_count": int(np.sum(finite < 0)),
        "zero_count": int(np.sum(finite == 0)),
    }


def cadence_summary(source: pd.Series) -> dict[str, Any]:
    timestamps = pd.to_datetime(source, errors="coerce")
    valid = pd.DatetimeIndex(timestamps.dropna())
    sorted_unique = valid.drop_duplicates().sort_values()
    deltas = sorted_unique.to_series().diff().dropna().dt.total_seconds().to_numpy() / 60.0
    unique, counts = np.unique(deltas, return_counts=True) if len(deltas) else ([], [])
    order = np.argsort(counts)[::-1] if len(deltas) else []
    distribution = [
        {"minutes": float(unique[index]), "count": int(counts[index])}
        for index in order[:10]
    ]
    return {
        "parse_failures": int(timestamps.isna().sum()),
        "duplicates": int(timestamps.duplicated().sum()),
        "monotonic_in_file": bool(timestamps.is_monotonic_increasing),
        "minimum": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "maximum": timestamps.max().isoformat() if timestamps.notna().any() else None,
        "dominant_interval_minutes": float(unique[order[0]]) if len(deltas) else None,
        "interval_distribution_top10": distribution,
    }


def infer_pwat_unit(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    medians = [item["median"] for item in summaries.values() if item["median"] is not None]
    q99_values = [item["q99"] for item in summaries.values() if item["q99"] is not None]
    negatives = sum(item["negative_count"] for item in summaries.values())
    if not medians:
        return {"inferred_unit": "unknown", "confidence": "none", "reason": "no finite PWAT values"}
    median = float(np.median(medians))
    q99 = float(np.max(q99_values))
    if negatives:
        return {
            "inferred_unit": "invalid",
            "confidence": "high",
            "reason": f"PWAT contains {negatives} negative values",
        }
    if median >= 2.0 and q99 <= 100.0:
        unit, confidence = "mm", "high"
        reason = "median is on a multi-millimetre scale and q99 is below 100 mm"
    elif median < 2.0 and q99 <= 10.0:
        unit, confidence = "cm_or_very_dry_mm", "low"
        reason = "values are small enough that metadata is required to distinguish cm from a very dry mm series"
    else:
        unit, confidence = "unknown", "low"
        reason = "distribution is outside the conservative PWAT mm/cm ranges"
    return {
        "inferred_unit": unit,
        "confidence": confidence,
        "reason": reason,
        "median_across_columns": median,
        "maximum_q99_across_columns": q99,
        "mm_to_cm_factor": 0.1,
        "converted_median_cm_if_mm": median * 0.1,
        "converted_maximum_q99_cm_if_mm": q99 * 0.1,
    }


def inspect(path: Path, expected_step: int, declared_pwat_unit: str, sample_rows: int) -> tuple[dict[str, Any], int]:
    resolved = path.expanduser().resolve()
    report: dict[str, Any] = {
        "path": str(resolved),
        "expected_step_minutes": expected_step,
        "declared_pwat_unit": declared_pwat_unit,
        "declared_pwat_to_cm": 0.1 if declared_pwat_unit == "mm" else 1.0,
        "problems": [],
    }
    problems: list[str] = report["problems"]
    if not resolved.is_file():
        problems.append(f"Parquet file does not exist: {resolved}")
        return report, 2

    parquet = pq.ParquetFile(resolved)
    metadata = parquet.metadata
    available = list(parquet.schema_arrow.names)
    missing = [column for column in EXPECTED_COLUMNS if column not in available]
    report["file"] = {
        "size_bytes": resolved.stat().st_size,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "created_by": metadata.created_by,
    }
    report["schema"] = {
        "columns": available,
        "missing_expected_columns": missing,
        "unexpected_columns": [column for column in available if column not in EXPECTED_COLUMNS],
        "arrow_types": {field.name: str(field.type) for field in parquet.schema_arrow},
    }
    if missing:
        problems.append(f"missing expected columns: {missing}")
    if metadata.num_rows == 0 or metadata.num_row_groups == 0:
        problems.append("Parquet contains no rows or row groups")
        return report, 2

    inspect_columns = [
        column for column in ("timestamp", "observe_power", *IRRADIANCE_COLUMNS, *PWAT_COLUMNS)
        if column in available
    ]
    frame = parquet.read(columns=inspect_columns).to_pandas()
    if "timestamp" in frame:
        report["timestamps"] = cadence_summary(frame["timestamp"])
        timestamps = report["timestamps"]
        if timestamps["parse_failures"]:
            problems.append(f"timestamp parse failures: {timestamps['parse_failures']}")
        if timestamps["duplicates"]:
            problems.append(f"duplicate timestamps: {timestamps['duplicates']}")
        dominant = timestamps["dominant_interval_minutes"]
        if dominant is None or not math.isclose(dominant, expected_step, abs_tol=1e-6):
            problems.append(
                f"time granularity mismatch: expected {expected_step} minutes, actual dominant interval {dominant}"
            )

    report["numeric"] = {
        column: numeric_summary(frame[column])
        for column in inspect_columns if column != "timestamp"
    }
    pwat_summaries = {column: report["numeric"][column] for column in PWAT_COLUMNS if column in frame}
    report["pwat_unit_check"] = infer_pwat_unit(pwat_summaries)
    inference = report["pwat_unit_check"]["inferred_unit"]
    if declared_pwat_unit == "mm" and inference not in {"mm", "cm_or_very_dry_mm"}:
        problems.append(f"PWAT values do not support the declared mm unit: inference={inference}")
    for column in IRRADIANCE_COLUMNS:
        if column in report["numeric"] and report["numeric"][column]["finite"] == 0:
            problems.append(f"{column} contains no finite supervision")
    if "observe_power" in report["numeric"] and report["numeric"]["observe_power"]["maximum"] is not None:
        maximum = report["numeric"]["observe_power"]["maximum"]
        if maximum > 468.0 * 1.05:
            problems.append(f"observe_power maximum {maximum} exceeds 468 MW by more than 5%")

    sample = parquet.read_row_group(0, columns=inspect_columns).slice(0, sample_rows).to_pandas()
    report["sample"] = json.loads(sample.to_json(orient="records", date_format="iso"))
    report["usable"] = not problems
    return report, 0 if not problems else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_step_minutes <= 0 or args.sample_rows <= 0:
        raise ValueError("expected step and sample rows must be positive")
    report, status = inspect(args.parquet, args.expected_step_minutes, args.pwat_unit, args.sample_rows)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_name(f".{args.output_json.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output_json)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
