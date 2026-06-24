from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate station-level prediction CSVs with per-station, pooled, and macro metrics."
    )
    parser.add_argument("--prediction-csvs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/pooled_evaluation")
    parser.add_argument("--dataset-names", nargs="*", default=None)
    return parser.parse_args()


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = pd.to_numeric(frame["prediction"], errors="coerce") - pd.to_numeric(frame["target"], errors="coerce")
    error = error.dropna()
    if len(error) == 0:
        return {"rows": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    return {
        "rows": int(len(error)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


def read_prediction_csv(path: Path, dataset_name: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"prediction", "target"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {missing}")
    out = frame.copy()
    out["dataset"] = dataset_name or path.stem
    out["prediction_csv"] = str(path)
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(item) for item in args.prediction_csvs]
    names = args.dataset_names or []
    if names and len(names) != len(paths):
        raise ValueError("--dataset-names length must match --prediction-csvs length")

    frames = [
        read_prediction_csv(path, names[index] if names else None)
        for index, path in enumerate(paths)
    ]
    combined = pd.concat(frames, ignore_index=True)

    per_rows = []
    for dataset, group in combined.groupby("dataset", dropna=False):
        row = {"dataset": dataset, **regression_metrics(group)}
        per_rows.append(row)
    per_station = pd.DataFrame(per_rows).sort_values("dataset").reset_index(drop=True)

    pooled = pd.DataFrame([{"scope": "pooled_all_rows", **regression_metrics(combined)}])
    macro = pd.DataFrame(
        [
            {
                "scope": "macro_average_stations",
                "num_datasets": int(len(per_station)),
                "mae": float(per_station["mae"].mean()),
                "rmse": float(per_station["rmse"].mean()),
                "bias": float(per_station["bias"].mean()),
                "rows_total": int(per_station["rows"].sum()),
            }
        ]
    )

    combined_path = output_dir / "pooled_predictions.csv"
    per_path = output_dir / "per_station_metrics.csv"
    pooled_path = output_dir / "pooled_metrics.csv"
    macro_path = output_dir / "macro_metrics.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    per_station.to_csv(per_path, index=False, encoding="utf-8-sig")
    pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")
    macro.to_csv(macro_path, index=False, encoding="utf-8-sig")

    print(f"per_station_metrics={per_path}")
    print(f"pooled_metrics={pooled_path}")
    print(f"macro_metrics={macro_path}")
    print(per_station.to_string(index=False))
    print(pooled.to_string(index=False))
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
