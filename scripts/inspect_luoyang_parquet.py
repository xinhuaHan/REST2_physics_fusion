"""Read-only schema and physical-field preflight for the new Luoyang Parquet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_PARQUET = Path(
    "/data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/"
    "Luoyang-Unified_format-V1-with_DNI_DHI.parquet"
)
REQUIRED_COLUMNS = (
    "timestamp", "final_power", "asi_path", "asi_path_timestamps", "GHI_mean_observe",
    "msl_forecast", "t2m_forecast", "u10_forecast", "v10_forecast", "u100_forecast", "v100_forecast",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the new Luoyang DNI/DHI Parquet without modifying it."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--expected-step-minutes", type=int, default=5)
    parser.add_argument("--rated-power", type=float, default=48629.73)
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def numeric_summary(source: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(source, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = values[np.isfinite(values)]
    return {
        "rows": int(len(source)),
        "finite": int(len(finite)),
        "missing": int(source.isna().sum()),
        "parse_failures": int(numeric.isna().sum() - source.isna().sum()),
        "minimum": float(np.min(finite)) if len(finite) else None,
        "q01": float(np.quantile(finite, 0.01)) if len(finite) else None,
        "q05": float(np.quantile(finite, 0.05)) if len(finite) else None,
        "median": float(np.quantile(finite, 0.50)) if len(finite) else None,
        "mean": float(np.mean(finite)) if len(finite) else None,
        "q95": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "q99": float(np.quantile(finite, 0.99)) if len(finite) else None,
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
    return {
        "parse_failures": int(timestamps.isna().sum()),
        "duplicates": int(timestamps.duplicated().sum()),
        "monotonic_in_file": bool(timestamps.is_monotonic_increasing),
        "minimum": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "maximum": timestamps.max().isoformat() if timestamps.notna().any() else None,
        "dominant_interval_minutes": float(unique[order[0]]) if len(deltas) else None,
        "interval_distribution_top10": [
            {"minutes": float(unique[index]), "count": int(counts[index])} for index in order[:10]
        ],
    }


def irradiance_candidates(columns: Sequence[str]) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {"ghi": [], "dni": [], "dhi": []}
    for column in columns:
        lowered = column.lower()
        for component in candidates:
            if component in lowered:
                candidates[component].append(column)
    return candidates


def inspect(path: Path, expected_step: int, rated_power: float, sample_rows: int) -> tuple[dict[str, Any], int]:
    resolved = path.expanduser().resolve()
    report: dict[str, Any] = {
        "path": str(resolved),
        "expected_step_minutes": expected_step,
        "rated_power": rated_power,
        "problems": [],
    }
    problems: list[str] = report["problems"]
    if not resolved.is_file():
        problems.append(f"Parquet file does not exist: {resolved}")
        return report, 2

    parquet = pq.ParquetFile(resolved)
    available = list(parquet.schema_arrow.names)
    missing = [column for column in REQUIRED_COLUMNS if column not in available]
    candidates = irradiance_candidates(available)
    report["file"] = {
        "size_bytes": resolved.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "created_by": parquet.metadata.created_by,
    }
    report["schema"] = {
        "columns": available,
        "missing_required_columns": missing,
        "irradiance_candidates": candidates,
        "arrow_types": {field.name: str(field.type) for field in parquet.schema_arrow},
    }
    if missing:
        problems.append(f"missing required columns: {missing}")
    for component in ("dni", "dhi"):
        if not candidates[component]:
            problems.append(f"no {component.upper()} column candidate found; do not enable its supervision")
    if parquet.metadata.num_rows == 0 or parquet.metadata.num_row_groups == 0:
        problems.append("Parquet contains no rows or row groups")
        return report, 2

    inspect_columns = list(dict.fromkeys([
        column for column in ("timestamp", "final_power", *sum(candidates.values(), [])) if column in available
    ]))
    frame = parquet.read(columns=inspect_columns).to_pandas()
    if "timestamp" in frame:
        report["timestamps"] = cadence_summary(frame["timestamp"])
        timestamp_report = report["timestamps"]
        if timestamp_report["parse_failures"]:
            problems.append(f"timestamp parse failures: {timestamp_report['parse_failures']}")
        if timestamp_report["duplicates"]:
            problems.append(f"duplicate timestamps: {timestamp_report['duplicates']}")
        dominant = timestamp_report["dominant_interval_minutes"]
        if dominant is None or not math.isclose(dominant, expected_step, abs_tol=1e-6):
            problems.append(
                f"time granularity mismatch: expected {expected_step} minutes, actual dominant interval {dominant}"
            )
    report["numeric"] = {
        column: numeric_summary(frame[column]) for column in inspect_columns if column != "timestamp"
    }
    for component, columns in candidates.items():
        for column in columns:
            if report["numeric"][column]["finite"] == 0:
                problems.append(f"{component.upper()} candidate {column} contains no finite values")
    if "final_power" in report["numeric"]:
        power = report["numeric"]["final_power"]
        report["target_quality"] = {
            "negative_values": power["negative_count"],
            "missing_values": power["missing"],
        }
        if power["maximum"] is not None and power["maximum"] > rated_power * 1.05:
            problems.append(f"final_power maximum {power['maximum']} exceeds rated power by more than 5%")
    sample = parquet.read_row_group(0, columns=inspect_columns).slice(0, sample_rows).to_pandas()
    report["sample"] = json.loads(sample.to_json(orient="records", date_format="iso"))
    report["usable"] = not problems
    return report, 0 if not problems else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_step_minutes <= 0 or args.rated_power <= 0 or args.sample_rows <= 0:
        raise ValueError("expected step, rated power and sample rows must be positive")
    report, status = inspect(args.parquet, args.expected_step_minutes, args.rated_power, args.sample_rows)
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
