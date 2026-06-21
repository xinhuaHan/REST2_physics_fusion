from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rest2_physics_fusion.data.preprocess import chronological_split, compute_norm_stats, read_training_csv
from rest2_physics_fusion.data.schema import DataSchema
from rest2_physics_fusion.data.windowing import PhysicsWindowDataset, collate_window_samples
from rest2_physics_fusion.models.serial_physics_model import SerialPhysicsForecaster
from rest2_physics_fusion.training.losses import regression_loss, regression_metrics


def _serialize_norm(stats: dict[str, object]) -> dict[str, list[float]]:
    return {
        "mean": [float(value) for value in stats["mean"]],
        "std": [float(value) for value in stats["std"]],
    }


def _deserialize_norm(stats: dict[str, list[float]] | None) -> dict[str, object] | None:
    if stats is None:
        return None
    import numpy as np

    return {
        "mean": np.asarray(stats["mean"], dtype=np.float32),
        "std": np.asarray(stats["std"], dtype=np.float32),
    }


def _load_checkpoint(path: str | Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_loaders(
    csv_path: str | Path,
    schema: DataSchema,
    window_steps: int,
    batch_size: int,
    serial_norm: dict[str, object] | None = None,
    physics_norm: dict[str, object] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, object], dict[str, object]]:
    frame = read_training_csv(csv_path, schema.timestamp_column)
    train_df, val_df, test_df = chronological_split(frame)
    serial_norm = serial_norm or compute_norm_stats(train_df, list(schema.serial_feature_columns))
    physics_norm = physics_norm or compute_norm_stats(train_df, list(schema.physics_feature_columns))

    def make_loader(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        dataset = PhysicsWindowDataset(df, schema, window_steps, serial_norm, physics_norm)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_window_samples)

    return make_loader(train_df, True), make_loader(val_df, False), make_loader(test_df, False), serial_norm, physics_norm


def train_model(
    csv_path: str | Path,
    *,
    output_dir: str | Path,
    schema: DataSchema | None = None,
    window_steps: int = 16,
    batch_size: int = 32,
    epochs: int = 3,
    lr: float = 1e-3,
    device: str = "cpu",
    serial_hidden: int = 64,
    weather_hidden: int = 32,
    physics_hidden: int = 64,
    fusion_hidden: int = 128,
    dropout: float = 0.1,
    use_rest2_calibration: bool = False,
    use_clear_sky_weather_head: bool = False,
    use_weather_prior_fusion: bool = False,
    use_clear_sky_power_prior: bool = False,
    weather_prior_weight_max: float = 1.0,
    prior_weight_l1: float = 0.0,
    sky_index_max: float = 2.0,
) -> Path:
    schema = schema or DataSchema()
    train_loader, val_loader, _, serial_norm, physics_norm = build_loaders(csv_path, schema, window_steps, batch_size)
    model_config = {
        "serial_input_dim": len(schema.serial_feature_columns),
        "weather_input_dim": len(schema.weather_feature_columns),
        "physics_input_dim": len(schema.physics_feature_columns),
        "serial_hidden": int(serial_hidden),
        "weather_hidden": int(weather_hidden),
        "physics_hidden": int(physics_hidden),
        "fusion_hidden": int(fusion_hidden),
        "dropout": float(dropout),
        "use_rest2_calibration": bool(use_rest2_calibration),
        "use_clear_sky_weather_head": bool(use_clear_sky_weather_head),
        "use_weather_prior_fusion": bool(use_weather_prior_fusion),
        "use_clear_sky_power_prior": bool(use_clear_sky_power_prior),
        "weather_prior_weight_max": float(weather_prior_weight_max),
        "prior_weight_l1": float(prior_weight_l1),
        "sky_index_max": float(sky_index_max),
        "target_column": schema.target_column,
        "physics_feature_columns": list(schema.physics_feature_columns),
        "physics_norm": _serialize_norm(physics_norm),
    }
    model = SerialPhysicsForecaster(
        serial_input_dim=len(schema.serial_feature_columns),
        weather_input_dim=len(schema.weather_feature_columns),
        physics_input_dim=len(schema.physics_feature_columns),
        serial_hidden=serial_hidden,
        weather_hidden=weather_hidden,
        physics_hidden=physics_hidden,
        fusion_hidden=fusion_hidden,
        dropout=dropout,
        use_rest2_calibration=use_rest2_calibration,
        use_clear_sky_weather_head=use_clear_sky_weather_head,
        use_weather_prior_fusion=use_weather_prior_fusion,
        use_clear_sky_power_prior=use_clear_sky_power_prior,
        weather_prior_weight_max=weather_prior_weight_max,
        sky_index_max=sky_index_max,
        target_column=schema.target_column,
        physics_feature_columns=tuple(schema.physics_feature_columns),
        physics_norm=_serialize_norm(physics_norm),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_path = output_path / "best.pt"
    best_val = float("inf")
    records: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch)
            loss = regression_loss(outputs["prediction"], batch["target"])
            if prior_weight_l1 > 0.0 and "prior_weight" in outputs:
                loss = loss + float(prior_weight_l1) * outputs["prior_weight"].mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch)
                val_losses.append(regression_loss(outputs["prediction"], batch["target"]).item())
        train_loss = float(sum(train_losses) / max(1, len(train_losses)))
        val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "schema": schema.__dict__,
                    "serial_norm": _serialize_norm(serial_norm),
                    "physics_norm": _serialize_norm(physics_norm),
                    "model_config": model_config,
                },
                best_path,
            )

    pd.DataFrame(records).to_csv(output_path / "epoch_losses.csv", index=False)
    return best_path


def evaluate_model(
    checkpoint_path: str | Path,
    csv_path: str | Path,
    *,
    schema: DataSchema | None = None,
    window_steps: int = 16,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, float]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    if schema is None:
        schema_state = checkpoint.get("schema", {})
        schema = DataSchema(target_column=schema_state.get("target_column", "target_ghi_5min"))
    serial_norm = _deserialize_norm(checkpoint.get("serial_norm"))
    physics_norm = _deserialize_norm(checkpoint.get("physics_norm"))
    _, _, test_loader, _, eval_physics_norm = build_loaders(
        csv_path,
        schema,
        window_steps,
        batch_size,
        serial_norm=serial_norm,
        physics_norm=physics_norm,
    )
    model_config = checkpoint.get("model_config", {})
    use_rest2_calibration = bool(model_config.get("use_rest2_calibration", False))
    calibration_norm = model_config.get("physics_norm") or _serialize_norm(eval_physics_norm)
    model = SerialPhysicsForecaster(
        serial_input_dim=len(schema.serial_feature_columns),
        weather_input_dim=int(model_config.get("weather_input_dim", len(schema.weather_feature_columns))),
        physics_input_dim=len(schema.physics_feature_columns),
        serial_hidden=int(model_config.get("serial_hidden", 64)),
        weather_hidden=int(model_config.get("weather_hidden", 32)),
        physics_hidden=int(model_config.get("physics_hidden", 64)),
        fusion_hidden=int(model_config.get("fusion_hidden", 128)),
        dropout=float(model_config.get("dropout", 0.1)),
        use_rest2_calibration=use_rest2_calibration,
        use_clear_sky_weather_head=bool(model_config.get("use_clear_sky_weather_head", False)),
        use_weather_prior_fusion=bool(model_config.get("use_weather_prior_fusion", False)),
        use_clear_sky_power_prior=bool(model_config.get("use_clear_sky_power_prior", False)),
        weather_prior_weight_max=float(model_config.get("weather_prior_weight_max", 1.0)),
        sky_index_max=float(model_config.get("sky_index_max", 2.0)),
        target_column=model_config.get("target_column", schema.target_column),
        physics_feature_columns=tuple(schema.physics_feature_columns),
        physics_norm=calibration_norm,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            predictions.append(outputs["prediction"])
            targets.append(batch["target"])
    pred = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    return regression_metrics(pred, target)
