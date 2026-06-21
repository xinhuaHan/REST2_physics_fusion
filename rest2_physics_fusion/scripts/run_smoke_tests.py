from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from test_data_windowing import test_window_dataset_shapes
from test_model_forward import test_model_forward_shape
from test_physics_features import test_build_physics_features_contains_contract


def main() -> None:
    test_build_physics_features_contains_contract()
    test_window_dataset_shapes()
    test_model_forward_shape()
    print("smoke_tests=passed")


if __name__ == "__main__":
    main()
