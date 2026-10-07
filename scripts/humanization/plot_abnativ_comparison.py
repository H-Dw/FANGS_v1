#!/usr/bin/env python3
"""Create the revised VHH2 AbNatiV comparison and quantitative summary."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tools.revision_common import COLORS, apply_style, save_figure, write_metadata


def bootstrap_mean_difference(left: np.ndarray, right: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    for index in range(n):
        values[index] = rng.choice(left, len(left), replace=True).mean() - rng.choice(right, len(right), replace=True).mean()
    return float(left.mean() - right.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def select_top_candidates(scores_dir: Path, n_selected: int) -> pd.DataFrame:
    """Keep the first n_selected original-score rows and join them to grafted scores."""
    if n_selected < 1:
        raise ValueError("n_selected must be a positive integer")
    original = pd.read_csv(scores_dir / "original_raw_embeddings.tsv", sep="\t")
    grafted = pd.read_csv(scores_dir / "grafted_raw_embeddings.tsv", sep="\t")
    original_columns = ["PDB_ID", "original_euclidean_all_raw_embeddings"]
    grafted_columns = ["PDB_ID", "grafted_euclidean_all_raw_embeddings"]
    if original.columns[:2].tolist() != original_columns or grafted.columns[:2].tolist() != grafted_columns:
        raise ValueError("Expected PDB_ID and whole-connector distance in the first two columns")
    if not original[original_columns[1]].is_monotonic_increasing:
        raise ValueError("Original score table must retain its ascending distance order")
    if not grafted[grafted_columns[1]].is_monotonic_increasing:
        raise ValueError("Grafted score table must retain its ascending distance order")
    if n_selected > len(original):
        raise ValueError(f"n_selected ({n_selected}) exceeds original score rows ({len(original)})")
    return grafted.iloc[:, :2].merge(
        original.iloc[:n_selected, :2], on="PDB_ID", sort=False, validate="one_to_one"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vh", type=Path)
    parser.add_argument("--vhh", type=Path)
    parser.add_argument("--top15", type=Path)
    parser.add_argument("--scores-dir", type=Path, help="Select candidates directly from original/grafted score tables")
    parser.add_argument("--n-selected", type=int, default=15, help="Number of lowest-distance original-score rows to retain")
    parser.add_argument("--source-table", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.n_selected < 1:
        parser.error("--n-selected must be a positive integer")
    if args.scores_dir is not None and (args.top15 is not None or args.source_table is not None):
        parser.error("Use --scores-dir, --top15, or --source-table as the candidate source")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_table is not None:
        plot_data = pd.read_csv(args.source_table, sep="\t")
        plot_data["both_scores_pass"] = plot_data["both_scores_pass"].astype(str).str.lower().eq("true")
    else:
        if args.vh is None or args.vhh is None or (args.top15 is None and args.scores_dir is None):
            raise RuntimeError("Provide --source-table or --vh/--vhh with --scores-dir or --top15")
        vh = pd.read_csv(args.vh)[["seq_id", "AbNatiV VH Score"]]
        vhh = pd.read_csv(args.vhh)[["seq_id", "AbNatiV VHH Score"]]
        scores = vh.merge(vhh, on="seq_id", validate="one_to_one")
        top = (
            select_top_candidates(args.scores_dir, args.n_selected)
            if args.scores_dir is not None else pd.read_csv(args.top15, sep="\t")
        )
        top_ids = set(top["PDB_ID"].astype(str))
        scores["PDB_ID"] = scores["seq_id"].str.replace(r"^FANGS_", "", regex=True)
        scores["group"] = np.select(
            [
                scores["PDB_ID"].isin(top_ids),
                scores["seq_id"].str.startswith("HuDiff"),
                scores["seq_id"].str.startswith("Original"),
                scores["seq_id"].str.startswith("TNF30"),
            ],
            ["FANGS", "HuDiff", "Original VHH2", "TNF30"],
            default="Other",
        )
        plot_data = scores[scores["group"].isin(["FANGS", "HuDiff", "Original VHH2", "TNF30"])].copy()
        plot_data["both_scores_pass"] = (
            (plot_data["AbNatiV VH Score"] >= 0.8) & (plot_data["AbNatiV VHH Score"] >= 0.8)
        )
    fangs = plot_data[plot_data["group"] == "FANGS"]
    hudiff = plot_data[plot_data["group"] == "HuDiff"]
    if len(fangs) != 15 or int(fangs["both_scores_pass"].sum()) != 9:
        raise ValueError(f"Expected FANGS dual-threshold result 9/15, found {int(fangs['both_scores_pass'].sum())}/{len(fangs)}")
    if args.source_table is None:
        top.to_csv(args.output_dir / "Fig3d_abnativ_candidates.tsv", sep="\t", index=False)

    summary = []
    for metric in ("AbNatiV VH Score", "AbNatiV VHH Score"):
        difference, low, high = bootstrap_mean_difference(
            fangs[metric].to_numpy(), hudiff[metric].to_numpy(), args.n_resamples, args.seed
        )
        summary.append(
            {
                "metric": metric,
                "FANGS_n": len(fangs),
                "FANGS_mean": fangs[metric].mean(),
                "HuDiff_n": len(hudiff),
                "HuDiff_mean": hudiff[metric].mean(),
                "FANGS_minus_HuDiff_mean_difference": difference,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    pd.DataFrame(summary).to_csv(args.output_dir / "Fig3d_abnativ_stats.tsv", sep="\t", index=False)
    plot_data.to_csv(args.output_dir / "Fig3d_abnativ_source.tsv", sep="\t", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(5.20, 3.55))
    groups = {
        "HuDiff": (COLORS["ochre"], "o"),
        "Original VHH2": (COLORS["gray"], "D"),
        "TNF30": (COLORS["lavender"], "s"),
    }
    baseline_handles = []
    for group, (color, marker) in groups.items():
        subset = plot_data[plot_data["group"] == group]
        handle = ax.scatter(
            subset["AbNatiV VHH Score"], subset["AbNatiV VH Score"],
            s=32, color=color, marker=marker, edgecolor="white", linewidth=0.5, label=group, zorder=3,
        )
        baseline_handles.append(handle)
    passed = fangs[fangs["both_scores_pass"]]
    failed = fangs[~fangs["both_scores_pass"]]
    passed_handle = ax.scatter(
        passed["AbNatiV VHH Score"], passed["AbNatiV VH Score"],
        s=34, color=COLORS["fangs"], edgecolor="white", linewidth=0.5,
        label="FANGS, both scores ≥ 0.80, 9/15", zorder=4,
    )
    failed_handle = ax.scatter(
        failed["AbNatiV VHH Score"], failed["AbNatiV VH Score"],
        s=34, color=COLORS["fangs_light"], edgecolor="white", linewidth=0.5,
        label="FANGS below either threshold", zorder=2,
    )
    ax.axvline(0.8, color=COLORS["gray"], linewidth=0.8, linestyle="--")
    ax.axhline(0.8, color=COLORS["gray"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("AbNatiV VHH score")
    ax.set_ylabel("AbNatiV VH score")
    ax.set_xlim(0.68, 0.93)
    ax.set_ylim(0.74, 0.92)
    fig.legend(
        [passed_handle, failed_handle],
        ["FANGS, both scores ≥ 0.80, 9/15", "FANGS below either threshold"],
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.legend(
        baseline_handles,
        ["HuDiff", "Original VHH2", "TNF30"],
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        columnspacing=1.4,
        handletextpad=0.4,
    )
    ax.set_box_aspect(1)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.72)
    save_figure(fig, args.output_dir / "Fig3d_abnativ_vh_vhh")
    plt.close(fig)
    write_metadata(
        args.output_dir / "Fig3d_abnativ_metadata.json",
        {
            "threshold_rule": "VH score and VHH score are each at least 0.80",
            "FANGS_success": "9/15",
            "n_selected": args.n_selected,
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "candidate_selection": (
                f"first {args.n_selected} original-score rows merged with grafted scores"
                if args.scores_dir is not None else "supplied candidate or figure-source table"
            ),
            "legend_placement": "two rows above scatter panel",
            "legend_row_1": "FANGS pass and FANGS below either threshold",
            "legend_row_2": "HuDiff, Original VHH2 and TNF30",
        },
    )


if __name__ == "__main__":
    main()
