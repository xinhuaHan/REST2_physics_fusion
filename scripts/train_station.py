from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_physics_moe import load_config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_experiment(config_path: Path, data_root: Path) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    experiment = raw.get("experiment", {})
    if not bool(experiment.get("fixed_year_split", False)):
        return
    expected_train = [int(year) for year in experiment.get("training_years", [])]
    expected_holdout = [int(year) for year in experiment.get("holdout_years", [])]
    metadata_path = data_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"processed metadata does not exist: {metadata_path}; run prepare_station_dataset.py first"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    problems: list[str] = []
    if metadata.get("training_years") != expected_train:
        problems.append(f"training_years={metadata.get('training_years')} expected={expected_train}")
    if metadata.get("holdout_years") != expected_holdout:
        problems.append(f"holdout_years={metadata.get('holdout_years')} expected={expected_holdout}")
    if metadata.get("normalization_fit_years") != expected_train:
        problems.append(
            f"normalization_fit_years={metadata.get('normalization_fit_years')} expected={expected_train}"
        )
    if metadata.get("normalization_fit_split") != "train":
        problems.append("normalization_fit_split is not 'train'")
    if metadata.get("client_holdout_locked") is not True:
        problems.append("client_holdout_locked is not true")
    if problems:
        details = "; ".join(problems)
        raise RuntimeError(
            "refusing to train because processed data does not satisfy the fixed-year experiment: "
            f"{details}. Re-run scripts/prepare_station_dataset.py with the current configuration."
        )
    print(
        f"year-split audit passed: train/normalization={expected_train}, client holdout={expected_holdout}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(ROOT / "configs" / "server_station.yaml"))
    parser.add_argument("--data")
    parser.add_argument("--smoke", action="store_true")
    args, _ = parser.parse_known_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    data_root = resolve_path(args.data or config.data.processed_dir)
    verify_experiment(config_path, data_root)
    command = [sys.executable, str(ROOT / "scripts" / "train_station_core.py"), *sys.argv[1:]]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
