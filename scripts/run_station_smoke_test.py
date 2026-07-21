from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
test_path = ROOT / "tests" / "test_station_server_pipeline.py"
spec = importlib.util.spec_from_file_location("station_server_test", test_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {test_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix="pv_station_smoke_") as directory:
    temp_root = Path(directory)
    module.test_five_csv_server_pipeline_is_two_modal_and_five_horizon(temp_root)
    config_path = temp_root / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["training"]["output_dir"] = str(temp_root / "outputs")
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "train_station.py"),
        "--config", str(config_path), "--data", str(temp_root / "processed"), "--smoke",
    ], check=True)
    if not (temp_root / "outputs" / "checkpoint_smoke.pt").exists():
        raise AssertionError("station smoke training did not save its checkpoint")
print("PASS five-table/five-horizon preprocessing, two-modal fusion, backward training and PCD closure")
