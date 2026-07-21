from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_physics_moe.physics.rest2 import PHYSICS_FEATURES, Rest2FeatureBuilder
from pv_physics_moe.physics.solar_geometry import solar_geometry


def column(frame: pd.DataFrame, name: str, default: float) -> torch.Tensor:
    if name not in frame:
        return torch.full((len(frame), 1), float(default))
    return torch.tensor(pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(), dtype=torch.float32)[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the standalone 26-column REST2 physical vector CSV")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--ghi-column", default="input_ghi")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if args.timestamp_column not in frame or args.ghi_column not in frame:
        raise ValueError("input CSV must contain timestamp and GHI columns")
    geometry = solar_geometry(frame[args.timestamp_column], args.latitude, args.longitude, args.timezone)
    inputs = {
        "mu0": torch.tensor(geometry["mu0"].to_numpy(), dtype=torch.float32)[:, None],
        "input_ghi": column(frame, args.ghi_column, 0.0),
        "dni_extra": torch.tensor(geometry["dni_extra"].to_numpy(), dtype=torch.float32)[:, None],
        "pressure_pa": column(frame, "pressure_pa", 101325.0),
        "pwv_cm": column(frame, "pwv_cm", 1.5),
        "aod700": column(frame, "aod700", 0.08),
        "precip": column(frame, "precip", 0.0),
        "wind_speed": column(frame, "wind_speed", 0.0),
        "weather_available": column(frame, "weather_is_joined", 1.0),
    }
    with torch.no_grad():
        vectors = Rest2FeatureBuilder()(inputs)[:, 0].cpu().numpy()
    output = pd.DataFrame(vectors, columns=PHYSICS_FEATURES)
    output.insert(0, "timestamp", frame[args.timestamp_column].astype(str).to_numpy())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"rows={len(output)} features={len(PHYSICS_FEATURES)} output={destination}")


if __name__ == "__main__":
    main()

