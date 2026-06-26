from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PNG visualizations from training, ablation, and prediction outputs.")
    parser.add_argument("--output-root", required=True, help="Pipeline/checkpoint output directory.")
    parser.add_argument("--prediction-csvs", nargs="*", default=None, help="Optional prediction CSVs exported by export_predictions.py.")
    parser.add_argument("--pooled-dir", default=None, help="Optional directory containing pooled/macro evaluation CSVs.")
    parser.add_argument("--fig-dir", default=None, help="Figure output directory. Default: <output-root>/figures")
    parser.add_argument("--max-prediction-points", type=int, default=500)
    return parser.parse_args()


def ensure_fig_dir(args: argparse.Namespace) -> Path:
    fig_dir = Path(args.fig_dir) if args.fig_dir else Path(args.output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"figure={path}")


def plot_epoch_losses(output_root: Path, fig_dir: Path) -> None:
    files = sorted(output_root.rglob("epoch_losses.csv"))
    rows = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        parts = path.relative_to(output_root).parts
        label = "/".join(parts[:-1]) if len(parts) > 1 else path.parent.name
        frame["run"] = label
        rows.append(frame)
    if not rows:
        return
    data = pd.concat(rows, ignore_index=True)
    for loss_column in ["train_loss", "val_loss"]:
        if loss_column not in data.columns:
            continue
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=data, x="epoch", y=loss_column, hue="run", marker="o")
        plt.title(loss_column)
        plt.xlabel("Epoch")
        plt.ylabel(loss_column)
        plt.legend(fontsize=7, loc="best")
        savefig(fig_dir / f"{loss_column}.png")


def plot_ablation(output_root: Path, fig_dir: Path) -> None:
    summary_path = output_root / "ablation_summary.csv"
    decisions_path = output_root / "ablation_decisions.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if {"dataset", "target_column", "comparison_model_type", "delta_mae_mean"}.issubset(summary.columns):
            data = summary.copy()
            data["mae_gain"] = -pd.to_numeric(data["delta_mae_mean"], errors="coerce")
            data["case"] = data["dataset"].astype(str) + "\n" + data["target_column"].astype(str)
            plt.figure(figsize=(12, max(5, 0.45 * len(data))))
            sns.barplot(data=data, y="case", x="mae_gain", hue="comparison_model_type")
            plt.axvline(0.0, color="black", linewidth=1)
            plt.title("MAE gain vs baseline")
            plt.xlabel("Positive means candidate improves over baseline")
            plt.ylabel("Dataset / Target")
            savefig(fig_dir / "ablation_mae_gain.png")
        if {"dataset", "target_column", "comparison_model_type", "delta_nmae_mean"}.issubset(summary.columns):
            data = summary.copy()
            data["nmae_gain"] = -pd.to_numeric(data["delta_nmae_mean"], errors="coerce")
            data["case"] = data["dataset"].astype(str) + "\n" + data["target_column"].astype(str)
            plt.figure(figsize=(12, max(5, 0.45 * len(data))))
            sns.barplot(data=data, y="case", x="nmae_gain", hue="comparison_model_type")
            plt.axvline(0.0, color="black", linewidth=1)
            plt.title("nMAE gain vs baseline")
            plt.xlabel("Positive means candidate improves over baseline")
            plt.ylabel("Dataset / Target")
            savefig(fig_dir / "ablation_nmae_gain.png")
    if decisions_path.exists():
        decisions = pd.read_csv(decisions_path)
        if "decision" in decisions.columns:
            counts = decisions["decision"].value_counts().reset_index()
            counts.columns = ["decision", "count"]
            plt.figure(figsize=(7, 4))
            sns.barplot(data=counts, x="decision", y="count")
            plt.title("Ablation decision counts")
            plt.xlabel("Decision")
            plt.ylabel("Count")
            plt.xticks(rotation=20)
            savefig(fig_dir / "decision_counts.png")


def plot_predictions(prediction_csvs: list[str], fig_dir: Path, max_points: int) -> None:
    for csv in prediction_csvs:
        path = Path(csv)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"prediction", "target"}
        if not required.issubset(frame.columns):
            continue
        plot_frame = frame.head(max_points).copy()
        x = pd.to_datetime(plot_frame["dtime"], errors="coerce") if "dtime" in plot_frame.columns else plot_frame.index
        plt.figure(figsize=(12, 5))
        plt.plot(x, plot_frame["target"], label="target", linewidth=1.5)
        plt.plot(x, plot_frame["prediction"], label="prediction", linewidth=1.2)
        if "clear_sky_base" in plot_frame.columns:
            plt.plot(x, plot_frame["clear_sky_base"], label="clear_sky", linewidth=1.0, alpha=0.7)
        plt.title(f"Prediction trace: {path.stem}")
        plt.xlabel("Time" if "dtime" in plot_frame.columns else "Index")
        plt.ylabel("GHI / target")
        plt.legend()
        savefig(fig_dir / f"{path.stem}_trace.png")

        error = pd.to_numeric(frame["prediction"], errors="coerce") - pd.to_numeric(frame["target"], errors="coerce")
        plt.figure(figsize=(8, 4))
        sns.histplot(error.dropna(), bins=50, kde=True)
        plt.title(f"Prediction error distribution: {path.stem}")
        plt.xlabel("prediction - target")
        savefig(fig_dir / f"{path.stem}_error_hist.png")


def plot_pooled(pooled_dir: Path | None, fig_dir: Path) -> None:
    if pooled_dir is None or not pooled_dir.exists():
        return
    per_path = pooled_dir / "per_station_metrics.csv"
    if per_path.exists():
        per = pd.read_csv(per_path)
        metrics = [column for column in ["mae", "rmse", "nmae", "nrmse"] if column in per.columns]
        for metric in metrics:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=per, x="dataset", y=metric)
            plt.title(f"Per-station {metric}")
            plt.xlabel("Dataset")
            plt.ylabel(metric)
            plt.xticks(rotation=20)
            savefig(fig_dir / f"per_station_{metric}.png")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    fig_dir = ensure_fig_dir(args)
    sns.set_theme(style="whitegrid")
    plot_epoch_losses(output_root, fig_dir)
    plot_ablation(output_root, fig_dir)
    plot_predictions(args.prediction_csvs or [], fig_dir, args.max_prediction_points)
    plot_pooled(Path(args.pooled_dir) if args.pooled_dir else None, fig_dir)


if __name__ == "__main__":
    main()
