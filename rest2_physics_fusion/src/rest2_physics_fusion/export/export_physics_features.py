from __future__ import annotations

from pathlib import Path

import pandas as pd

from rest2_physics_fusion.physics.physics_features import PhysicsConfig, build_physics_features


def export_excel_physics_csv(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    file_name: str,
    output_name: str,
    timestamp_column: str,
    ghi_column: str,
    source_type: str,
    station_name: str,
    config: PhysicsConfig,
) -> Path:
    input_path = Path(input_dir) / file_name
    output_path = Path(output_dir) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(input_path)
    physics = build_physics_features(
        frame,
        timestamp_column=timestamp_column,
        ghi_column=ghi_column,
        config=config,
        source_type=source_type,
        station_name=station_name,
    )
    physics["source_file"] = file_name
    physics.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
