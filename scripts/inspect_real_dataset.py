from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def inspect_csv(path: Path) -> dict:
    frame = pd.read_csv(path)
    timestamp_col = "timeStamp" if "timeStamp" in frame.columns else "timestamp"
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    numeric = frame.drop(columns=[timestamp_col]).apply(pd.to_numeric, errors="coerce")
    diffs = timestamps.sort_values().diff().dropna().dt.total_seconds()
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "start": None if timestamps.notna().sum() == 0 else str(timestamps.min()),
        "end": None if timestamps.notna().sum() == 0 else str(timestamps.max()),
        "bad_timestamps": int(timestamps.isna().sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "interval_seconds_top": {str(float(key)): int(value) for key, value in diffs.value_counts().head(8).items()},
        "missing": {name: int(value) for name, value in numeric.isna().sum().items()},
        "min": {name: float(value) for name, value in numeric.min().items()},
        "max": {name: float(value) for name, value in numeric.max().items()},
        "negative": {name: int((numeric[name] < 0).sum()) for name in numeric},
    }


def parse_image_timestamp(path: Path) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(path.stem, format="%Y%m%d_%H%M%S")
    except ValueError:
        return None


def inspect_images(root: Path) -> dict:
    counts = Counter()
    total, bad_names = 0, 0
    first_time = last_time = None
    samples: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = Path(dirpath) / filename
            timestamp = parse_image_timestamp(path)
            if timestamp is None:
                bad_names += 1
                continue
            total += 1
            counts[str(timestamp.year)] += 1
            if first_time is None or timestamp < first_time:
                first_time = timestamp
            if last_time is None or timestamp > last_time:
                last_time = timestamp
            if len(samples) < 5:
                samples.append(path)
    dimensions = []
    for path in samples:
        with Image.open(path) as image:
            dimensions.append({"path": str(path), "size": list(image.size), "mode": image.mode})
    return {
        "count": total,
        "counts_by_year": dict(sorted(counts.items())),
        "bad_filenames": bad_names,
        "start": None if first_time is None else str(first_time),
        "end": None if last_time is None else str(last_time),
        "sample_dimensions": dimensions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=r"C:\Users\ADMIN\Desktop\dataset")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.dataset_root)
    report = {
        "irradiance": inspect_csv(root / "Folsom_irradiance.csv"),
        "weather": inspect_csv(root / "Folsom_weather.csv"),
        "images": inspect_images(root),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
