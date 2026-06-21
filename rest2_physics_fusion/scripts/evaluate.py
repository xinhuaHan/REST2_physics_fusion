from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.training.train import evaluate_model
from rest2_physics_fusion.data.schema import DataSchema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--window-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-column", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = DataSchema(target_column=args.target_column) if args.target_column else None
    metrics = evaluate_model(
        args.checkpoint,
        args.csv,
        schema=schema,
        window_steps=args.window_steps,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(metrics)


if __name__ == "__main__":
    main()
