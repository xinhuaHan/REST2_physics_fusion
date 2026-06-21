from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rest2_physics_fusion.training.model_selection import MODEL_VARIANTS, resolve_model_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions-csv", default=str(ROOT / "outputs" / "ablation_weather_prior" / "ablation_decisions.csv"))
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-yaml", default=None)
    parser.add_argument("--default-model-type", default="baseline", choices=sorted(MODEL_VARIANTS))
    parser.add_argument(
        "--allowed-model-types",
        nargs="*",
        default=[
            "physics_feature_only",
            "clear_sky_power_prior",
            "weather_prior_weak",
            "rest2_calibrated",
            "weather_prior",
            "weather_prior_rest2",
        ],
        choices=sorted(MODEL_VARIANTS),
        help="Candidate model types allowed for automatic promotion. Only promote_candidate rows are selected by default.",
    )
    parser.add_argument(
        "--include-watch",
        action="store_true",
        help="Allow watch_high_variance rows to be selected. By default only promote_candidate rows are selected.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    cwd_candidate = Path.cwd() / value
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate
    return ROOT / value


def build_selection(
    decisions: pd.DataFrame,
    *,
    default_model_type: str,
    allowed_model_types: set[str],
    include_watch: bool,
) -> pd.DataFrame:
    required = {"dataset", "target_column", "comparison_model_type", "decision", "mae_gain"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"Decisions CSV is missing columns: {missing}")
    resolve_model_variant(default_model_type)
    for model_type in allowed_model_types:
        resolve_model_variant(model_type)

    allowed_decisions = {"promote_candidate"}
    if include_watch:
        allowed_decisions.add("watch_high_variance")

    rows = []
    for (dataset, target_column), group in decisions.groupby(["dataset", "target_column"], dropna=False):
        eligible = group[
            group["comparison_model_type"].isin(allowed_model_types)
            & group["decision"].isin(allowed_decisions)
        ].copy()
        if eligible.empty:
            selected = default_model_type
            source = None
            reason = "default_no_promoted_candidate"
            if "candidate_win_rate_mae" not in group.columns:
                group = group.copy()
                group["candidate_win_rate_mae"] = 0.0
            best_rejected = group.sort_values(["mae_gain", "candidate_win_rate_mae"], ascending=[False, False]).iloc[0]
            detail = (
                f"no eligible promote_candidate; best_candidate={best_rejected['comparison_model_type']} "
                f"decision={best_rejected['decision']} mae_gain={float(best_rejected['mae_gain']):.6f}"
            )
        else:
            if "candidate_win_rate_mae" not in eligible.columns:
                eligible["candidate_win_rate_mae"] = 0.0
            eligible = eligible.sort_values(
                ["decision", "mae_gain", "candidate_win_rate_mae"],
                ascending=[True, False, False],
            )
            source = eligible.iloc[0]
            selected = str(source["comparison_model_type"])
            reason = str(source["decision"])
            detail = "selected_promoted_candidate"
        row = {
            "dataset": dataset,
            "target_column": target_column,
            "selected_model_type": selected,
            "source_comparison_model_type": selected if source is not None else "",
            "candidate_selected": bool(source is not None),
            "decision": str(source["decision"]) if source is not None else "default",
            "mae_gain": float(source["mae_gain"]) if source is not None else 0.0,
            "rmse_gain": float(source.get("rmse_gain", 0.0)) if source is not None else 0.0,
            "candidate_win_rate_mae": float(source.get("candidate_win_rate_mae", 0.0)) if source is not None else 0.0,
            "selection_reason": reason,
            "selection_detail": detail,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "target_column"]).reset_index(drop=True)


def write_yaml(selection: pd.DataFrame, output_path: Path) -> None:
    lines = ["model_selection:"]
    for row in selection.to_dict(orient="records"):
        lines.extend(
            [
                f"  - dataset: {row['dataset']}",
                f"    target_column: {row['target_column']}",
                f"    selected_model_type: {row['selected_model_type']}",
                f"    candidate_selected: {str(bool(row.get('candidate_selected', False))).lower()}",
                f"    decision: {row['decision']}",
                f"    mae_gain: {float(row['mae_gain']):.6f}",
                f"    candidate_win_rate_mae: {float(row['candidate_win_rate_mae']):.6f}",
                f"    selection_reason: {row.get('selection_reason', '')}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    decisions_path = resolve_path(args.decisions_csv)
    output_csv = resolve_path(args.output_csv) if args.output_csv else decisions_path.parent / "model_selection.csv"
    output_yaml = resolve_path(args.output_yaml) if args.output_yaml else decisions_path.parent / "model_selection.yaml"
    decisions = pd.read_csv(decisions_path)
    selection = build_selection(
        decisions,
        default_model_type=args.default_model_type,
        allowed_model_types=set(args.allowed_model_types),
        include_watch=bool(args.include_watch),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_csv, index=False, encoding="utf-8-sig")
    write_yaml(selection, output_yaml)
    print(f"model_selection_csv={output_csv}")
    print(f"model_selection_yaml={output_yaml}")
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
