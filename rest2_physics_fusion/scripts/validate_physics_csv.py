from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "dtime",
    "source_type",
    "station_name",
    "input_ghi",
    "ghi_clear_target",
    "dni_clear_target",
    "dhi_clear_target",
    "mu0_target",
    "apparent_zenith",
    "dni_extra",
    "pressure_pa",
    "pwv_cm",
    "aod700",
    "t_clr_dd",
    "clear_sky_index_last",
    "smart_persistence_ghi",
    "feature_quality_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default=str(Path(__file__).resolve().parents[1] / "outputs" / "physics_csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in sorted(Path(args.csv_dir).glob("*.csv")):
        df = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        nan_counts = df[REQUIRED_COLUMNS].isna().sum()
        problematic = nan_counts[nan_counts > 0].to_dict()
        print(f"{path.name}: rows={len(df)} required_column_nans={problematic or 'none'}")


if __name__ == "__main__":
    main()
