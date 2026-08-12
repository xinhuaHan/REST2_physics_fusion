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
    "timestamp", "final_power", "GHI-onsite", "GHI-onsite-timestamps",
    "estimated_DNI-onsite", "estimated_DNI-onsite-timestamps",
    "estimated_DHI-onsite", "estimated_DHI-onsite-timestamps",
    "asi_path-onsite", "asi_path-onsite-timestamps",
    "GHI_mean-NWP_forecast", "msl-NWP_forecast", "t2m-NWP_forecast",
    "u10-NWP_forecast", "v10-NWP_forecast", "u100-NWP_forecast", "v100-NWP_forecast",
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
        if lowered.endswith("-timestamps"):
            continue
        for component in candidates:
            if component in lowered:
                candidates[component].append(column)
    return candidates


def _as_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    return [value]


def list_numeric_summary(source: pd.Series) -> dict[str, Any]:
    lengths, flattened, malformed_rows = [], [], 0
    rows_with_finite = 0
    for value in source:
        values = _as_list(value)
        lengths.append(len(values))
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(np.float64)
        if len(values) and not np.isfinite(numeric).any() and any(item is not None for item in values):
            malformed_rows += 1
        rows_with_finite += int(np.isfinite(numeric).any())
        flattened.extend(numeric.tolist())
    array = np.asarray(flattened, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "container": "list",
        "rows": int(len(source)),
        "rows_with_nonempty_list": int(sum(length > 0 for length in lengths)),
        "rows_with_finite_value": rows_with_finite,
        "list_length_minimum": int(min(lengths)) if lengths else 0,
        "list_length_median": float(np.median(lengths)) if lengths else 0.0,
        "list_length_maximum": int(max(lengths)) if lengths else 0,
        "elements": int(len(array)),
        "finite": int(len(finite)),
        "missing_or_unparseable_elements": int(len(array) - len(finite)),
        "malformed_rows": malformed_rows,
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


def list_timestamp_alignment(values: pd.Series, timestamps: pd.Series, issues: pd.Series) -> dict[str, Any]:
    offsets, timestamp_parse_failures = [], 0
    length_mismatches = 0
    rows_with_future_5_to_240 = 0
    for issue, value_list, timestamp_list in zip(issues, values, timestamps):
        values_for_row, stamps_for_row = _as_list(value_list), _as_list(timestamp_list)
        if len(values_for_row) != len(stamps_for_row):
            length_mismatches += 1
            continue
        issue_time = pd.Timestamp(issue)
        parsed = pd.to_datetime(pd.Series(stamps_for_row), errors="coerce")
        timestamp_parse_failures += int(parsed.isna().sum())
        valid = parsed.dropna()
        row_offsets = (valid - issue_time).dt.total_seconds().to_numpy() / 60.0
        offsets.extend(row_offsets.tolist())
        rows_with_future_5_to_240 += int(np.any((row_offsets >= 5.0) & (row_offsets <= 240.0)))
    finite = np.asarray(offsets, dtype=np.float64)
    return {
        "timestamp_column": str(timestamps.name),
        "length_mismatch_rows": length_mismatches,
        "timestamp_parse_failures": timestamp_parse_failures,
        "timestamp_elements": int(len(finite)),
        "offset_minutes_minimum": float(np.min(finite)) if len(finite) else None,
        "offset_minutes_median": float(np.median(finite)) if len(finite) else None,
        "offset_minutes_maximum": float(np.max(finite)) if len(finite) else None,
        "rows_with_a_timestamp_in_t_plus_5_to_t_plus_240": rows_with_future_5_to_240,
    }


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

    value_candidates = list(dict.fromkeys(sum(candidates.values(), [])))
    timestamp_columns = [f"{column}-timestamps" for column in value_candidates if f"{column}-timestamps" in available]
    inspect_columns = list(dict.fromkeys(["timestamp", "final_power", *value_candidates, *timestamp_columns]))
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
    list_columns = {
        field.name for field in parquet.schema_arrow if str(field.type).startswith("list<")
    }
    report["numeric"] = {
        column: (list_numeric_summary(frame[column]) if column in list_columns else numeric_summary(frame[column]))
        for column in value_candidates + (["final_power"] if "final_power" in frame else [])
    }
    report["list_timestamp_alignment"] = {
        column: list_timestamp_alignment(frame[column], frame[f"{column}-timestamps"], frame["timestamp"])
        for column in value_candidates
        if column in list_columns and f"{column}-timestamps" in frame
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
