from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.data.preprocess import chronological_split, compute_norm_stats, read_training_csv
from rest2_physics_fusion.data.schema import DataSchema, validate_columns


DEFAULT_INDEX_COLUMNS = (
    "dtime",
    "source_type",
    "station_name",
    "input_ghi",
    "target_ghi_5min",
    "target_ghi_4h",
    "target_ghi_1d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export fixed-schema physics features from a model_ready CSV as vectors "
            "for external fusion models such as PVMMOE."
        )
    )
    parser.add_argument("--csv", required=True, help="Input model_ready CSV.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "physics_vectors"))
    parser.add_argument("--target-column", default="target_ghi_5min")
    parser.add_argument("--prefix", default=None, help="Output file prefix. Default: input CSV stem.")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Which chronological split to export. Normalization stats are always fit on train.",
    )
    parser.add_argument("--window-steps", type=int, default=0, help="Keep rows after this warmup offset.")
    parser.add_argument("--normalize", action="store_true", help="Also export normalized physics vectors.")
    parser.add_argument("--write-csv", action="store_true", help="Also export a wide CSV with physics columns.")
    parser.add_argument("--index-columns", nargs="*", default=list(DEFAULT_INDEX_COLUMNS))
    return parser.parse_args()


def select_split(frame: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train_df, val_df, test_df = chronological_split(frame)
    if split == "train":
        return train_df, train_df, "train"
    if split == "val":
        return val_df, train_df, "val"
    if split == "test":
        return test_df, train_df, "test"
    return frame, train_df, "all"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or csv_path.stem

    schema = DataSchema(target_column=args.target_column)
    frame = read_training_csv(csv_path, schema.timestamp_column)
    validate_columns(set(frame.columns), schema)
    export_frame, train_frame, split_name = select_split(frame, args.split)
    if args.window_steps > 0:
        export_frame = export_frame.iloc[int(args.window_steps) :].reset_index(drop=True)
    physics_columns = list(schema.physics_feature_columns)

    physics = export_frame[physics_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    physics = np.nan_to_num(physics, nan=0.0, posinf=0.0, neginf=0.0)
    vector_path = output_dir / f"{prefix}_{split_name}_physics_vectors.npy"
    np.save(vector_path, physics)

    normalized_path = None
    norm_stats_path = None
    if args.normalize:
        stats = compute_norm_stats(train_frame, physics_columns)
        normalized = (physics - stats["mean"]) / stats["std"]
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        normalized_path = output_dir / f"{prefix}_{split_name}_physics_vectors_normalized.npy"
        norm_stats_path = output_dir / f"{prefix}_physics_norm.json"
        np.save(normalized_path, normalized)
        write_json(
            norm_stats_path,
            {
                "columns": physics_columns,
                "mean": [float(value) for value in stats["mean"]],
                "std": [float(value) for value in stats["std"]],
                "fit_split": "train",
            },
        )

    index_columns = [column for column in args.index_columns if column in export_frame.columns]
    index_frame = export_frame[index_columns].copy()
    index_path = output_dir / f"{prefix}_{split_name}_physics_index.csv"
    index_frame.to_csv(index_path, index=False, encoding="utf-8-sig")

    columns_path = output_dir / f"{prefix}_physics_columns.txt"
    columns_path.write_text("\n".join(physics_columns) + "\n", encoding="utf-8")

    csv_vector_path = None
    if args.write_csv:
        csv_vector_path = output_dir / f"{prefix}_{split_name}_physics_vectors.csv"
        export_frame[physics_columns].to_csv(csv_vector_path, index=False, encoding="utf-8-sig")

    metadata = {
        "source_csv": str(csv_path),
        "split": split_name,
        "target_column": args.target_column,
        "timestamp_column": schema.timestamp_column,
        "rows": int(physics.shape[0]),
        "physics_dim": int(physics.shape[1]),
        "physics_columns": physics_columns,
        "vector_path": str(vector_path),
        "index_path": str(index_path),
        "columns_path": str(columns_path),
        "normalized_vector_path": str(normalized_path) if normalized_path else "",
        "norm_stats_path": str(norm_stats_path) if norm_stats_path else "",
        "csv_vector_path": str(csv_vector_path) if csv_vector_path else "",
        "note": "Rows in vectors and index CSV are aligned by row order.",
    }
    metadata_path = output_dir / f"{prefix}_{split_name}_physics_metadata.json"
    write_json(metadata_path, metadata)

    print(f"vector_npy={vector_path}")
    print(f"index_csv={index_path}")
    print(f"metadata_json={metadata_path}")
    print(f"rows={physics.shape[0]} physics_dim={physics.shape[1]}")
    if normalized_path:
        print(f"normalized_vector_npy={normalized_path}")
    if csv_vector_path:
        print(f"vector_csv={csv_vector_path}")


if __name__ == "__main__":
    main()
