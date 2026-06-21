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
    parser.add_argument("--summary-csv", default=str(ROOT / "outputs" / "ablation" / "ablation_summary.csv"))
    parser.add_argument("--results-csv", default=str(ROOT / "outputs" / "ablation" / "ablation_results.csv"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-mae-gain", type=float, default=0.5)
    parser.add_argument("--min-mae-win-rate", type=float, default=0.6)
    parser.add_argument("--max-delta-std-ratio", type=float, default=1.5)
    parser.add_argument("--selection-output-csv", default=None)
    parser.add_argument("--selection-output-yaml", default=None)
    parser.add_argument("--selection-default-model-type", default="baseline", choices=sorted(MODEL_VARIANTS))
    parser.add_argument(
        "--selection-allowed-model-types",
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
    )
    parser.add_argument("--selection-include-watch", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    cwd_candidate = Path.cwd() / value
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate
    return ROOT / value


def classify_row(row: pd.Series, min_mae_gain: float, min_mae_win_rate: float, max_delta_std_ratio: float) -> str:
    mae_gain = -float(row["delta_mae_mean"])
    rmse_gain = -float(row["delta_rmse_mean"])
    win_rate = float(row.get("candidate_win_rate_mae", row.get("rest2_win_rate_mae", 0.0)))
    delta_std = float(row["delta_mae_std"])
    stable_enough = delta_std <= max(min_mae_gain * max_delta_std_ratio, abs(mae_gain) * max_delta_std_ratio)

    if mae_gain >= min_mae_gain and win_rate >= min_mae_win_rate and stable_enough:
        return "promote_candidate"
    if mae_gain >= min_mae_gain and win_rate >= min_mae_win_rate:
        return "watch_high_variance"
    if mae_gain > 0.0 or rmse_gain > 0.0:
        return "weak_or_mixed"
    return "reject"


def add_decisions(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = summary.copy()
    out["mae_gain"] = -out["delta_mae_mean"]
    out["rmse_gain"] = -out["delta_rmse_mean"]
    out["decision"] = out.apply(
        lambda row: classify_row(
            row,
            min_mae_gain=args.min_mae_gain,
            min_mae_win_rate=args.min_mae_win_rate,
            max_delta_std_ratio=args.max_delta_std_ratio,
        ),
        axis=1,
    )
    return out.sort_values(["decision", "mae_gain"], ascending=[True, False]).reset_index(drop=True)


def write_report(decisions: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> Path:
    counts = decisions["decision"].value_counts().to_dict()
    promote = decisions[decisions["decision"] == "promote_candidate"]
    watch = decisions[decisions["decision"] == "watch_high_variance"]
    report_path = output_dir / "ablation_decision_report.md"
    lines = [
        "# Physics Fusion Ablation Decision Report",
        "",
        "## Gate",
        "",
        f"- `min_mae_gain`: {args.min_mae_gain}",
        f"- `min_mae_win_rate`: {args.min_mae_win_rate}",
        f"- `max_delta_std_ratio`: {args.max_delta_std_ratio}",
        "",
        "## Decision Counts",
        "",
    ]
    for key in ("promote_candidate", "watch_high_variance", "weak_or_mixed", "reject"):
        lines.append(f"- `{key}`: {counts.get(key, 0)}")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
        ]
    )
    if promote.empty:
        lines.append("Do not enable any candidate physics branch as a global default yet. Keep candidates optional and continue diagnostics.")
    else:
        lines.append("Candidate physics branches can be promoted only for the listed dataset-target cases, not globally.")
    if not watch.empty:
        lines.append("High-variance candidates should be rerun on real data or more seeds before formalization.")
    lines.append("Rows marked `weak_or_mixed`, `watch_high_variance`, or `reject` are not selected by default; the safe fallback is `baseline`.")
    display_columns = [
        column
        for column in [
            "dataset",
            "target_column",
            "comparison_model_type",
            "mae_gain",
            "rmse_gain",
            "candidate_win_rate_mae",
            "rest2_win_rate_mae",
            "delta_mae_std",
            "decision",
        ]
        if column in decisions.columns
    ]
    lines.extend(["", "## Promote Candidates", ""])
    lines.append("```text")
    lines.append(promote[display_columns].to_string(index=False) if not promote.empty else "None.")
    lines.append("```")
    lines.extend(["", "## Watch Candidates", ""])
    lines.append("```text")
    lines.append(watch[display_columns].to_string(index=False) if not watch.empty else "None.")
    lines.append("```")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_model_selection(decisions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    resolve_model_variant(args.selection_default_model_type)
    allowed_model_types = set(args.selection_allowed_model_types)
    for model_type in allowed_model_types:
        resolve_model_variant(model_type)
    allowed_decisions = {"promote_candidate"}
    if args.selection_include_watch:
        allowed_decisions.add("watch_high_variance")

    rows = []
    for (dataset, target_column), group in decisions.groupby(["dataset", "target_column"], dropna=False):
        eligible = group[
            group["comparison_model_type"].isin(allowed_model_types)
            & group["decision"].isin(allowed_decisions)
        ].copy()
        if eligible.empty:
            source = None
            selected = args.selection_default_model_type
            reason = "default_no_promoted_candidate"
            best_rejected = group.sort_values(["mae_gain", "candidate_win_rate_mae"], ascending=[False, False]).iloc[0]
            rejection_detail = (
                f"no eligible promote_candidate; best_candidate={best_rejected['comparison_model_type']} "
                f"decision={best_rejected['decision']} mae_gain={float(best_rejected['mae_gain']):.6f}"
            )
        else:
            eligible = eligible.sort_values(
                ["decision", "mae_gain", "candidate_win_rate_mae"],
                ascending=[True, False, False],
            )
            source = eligible.iloc[0]
            selected = str(source["comparison_model_type"])
            reason = str(source["decision"])
            rejection_detail = "selected_promoted_candidate"
        rows.append(
            {
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
                "selection_detail": rejection_detail,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "target_column"]).reset_index(drop=True)


def write_model_selection_yaml(selection: pd.DataFrame, path: Path) -> None:
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_path = resolve_path(args.summary_csv)
    results_path = resolve_path(args.results_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {results_path}")

    output_dir = resolve_path(args.output_dir) if args.output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    decisions = add_decisions(summary, args)
    decisions_path = output_dir / "ablation_decisions.csv"
    candidates_path = output_dir / "promote_candidates.csv"
    watch_path = output_dir / "watch_candidates.csv"
    decisions.to_csv(decisions_path, index=False, encoding="utf-8-sig")
    decisions[decisions["decision"] == "promote_candidate"].to_csv(candidates_path, index=False, encoding="utf-8-sig")
    decisions[decisions["decision"] == "watch_high_variance"].to_csv(watch_path, index=False, encoding="utf-8-sig")
    decisions.to_csv(output_dir / "rest2_decisions.csv", index=False, encoding="utf-8-sig")
    decisions[decisions["decision"] == "promote_candidate"].to_csv(
        output_dir / "rest2_promote_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decisions[decisions["decision"] == "watch_high_variance"].to_csv(
        output_dir / "rest2_watch_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report_path = write_report(decisions, output_dir, args)
    selection = build_model_selection(decisions, args)
    selection_path = resolve_path(args.selection_output_csv) if args.selection_output_csv else output_dir / "model_selection.csv"
    selection_yaml_path = resolve_path(args.selection_output_yaml) if args.selection_output_yaml else output_dir / "model_selection.yaml"
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")
    write_model_selection_yaml(selection, selection_yaml_path)

    print(f"decisions_csv={decisions_path}")
    print(f"promote_candidates_csv={candidates_path}")
    print(f"watch_candidates_csv={watch_path}")
    print(f"decision_report={report_path}")
    print(f"model_selection_csv={selection_path}")
    print(f"model_selection_yaml={selection_yaml_path}")
    print_columns = [
        column
        for column in [
            "dataset",
            "target_column",
            "comparison_model_type",
            "mae_gain",
            "rmse_gain",
            "candidate_win_rate_mae",
            "rest2_win_rate_mae",
            "decision",
        ]
        if column in decisions.columns
    ]
    print(decisions[print_columns].to_string(index=False))


if __name__ == "__main__":
    main()
