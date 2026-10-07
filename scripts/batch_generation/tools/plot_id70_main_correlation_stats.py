#!/usr/bin/env python3
"""Replot the id70 Fig. 1g bars from an audited correlation table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SHARED_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_SHARED_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SHARED_TOOLS))

from revision_common import (
    COLORS,
    annotate_abs_spearman_bar,
    apply_style,
    place_legend_upper_right,
    save_figure,
    write_metadata,
)


METRICS = ["Connector", "Global structure", "Global sequence"]
OUTCOMES = [("cRMSD", COLORS["fangs"]), ("pTM", COLORS["blue"])]
REQUIRED = {
    "metric_label",
    "outcome_label",
    "rho",
    "abs_rho",
    "abs_ci_low",
    "abs_ci_high",
    "q_value",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stats = pd.read_csv(args.stats_table, sep="\t")
    missing = sorted(REQUIRED.difference(stats.columns))
    if missing:
        raise ValueError(f"Missing columns in statistics table: {missing}")
    keys = set(zip(stats["metric_label"], stats["outcome_label"]))
    expected = {(metric, outcome) for metric in METRICS for outcome, _ in OUTCOMES}
    if len(stats) != 6 or keys != expected or stats.duplicated(["metric_label", "outcome_label"]).any():
        raise ValueError("Expected exactly one row for every metric and outcome combination")
    if not np.allclose(stats["abs_rho"], stats["rho"].abs(), atol=1e-12):
        raise ValueError("abs_rho must equal the absolute value of signed rho")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(
        args.output_dir / "Fig1g_id70_virtual_grafting_comparison_source.tsv",
        sep="\t",
        index=False,
    )

    apply_style()
    fig, ax = plt.subplots(figsize=(4.7, 2.85))
    positions = np.arange(len(METRICS))
    width = 0.34
    for offset, (outcome, color) in zip((-width / 2, width / 2), OUTCOMES):
        rows = stats[stats["outcome_label"] == outcome].set_index("metric_label").loc[METRICS]
        values = rows["abs_rho"].to_numpy(dtype=float)
        lower = rows["abs_ci_low"].to_numpy(dtype=float)
        upper = rows["abs_ci_high"].to_numpy(dtype=float)
        bars = ax.bar(
            positions + offset,
            values,
            width=width,
            color=color,
            label=outcome,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.errorbar(
            positions + offset,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="none",
            ecolor=COLORS["gray"],
            capsize=2,
            linewidth=0.8,
        )
        for bar, (_, row) in zip(bars, rows.iterrows()):
            annotate_abs_spearman_bar(
                ax,
                bar,
                float(row["abs_rho"]),
                float(row["abs_ci_high"]),
                float(row["q_value"]),
            )
    ax.set_ylabel("|Spearman ρ|")
    ax.set_xticks(positions, METRICS)
    ax.set_ylim(0.0, 1.08)
    place_legend_upper_right(ax, ncol=1)
    fig.subplots_adjust(left=0.14, right=0.76, bottom=0.20, top=0.94)
    save_figure(fig, args.output_dir / "Fig1g_id70_virtual_grafting_comparison")
    plt.close(fig)
    write_metadata(
        args.output_dir / "Fig1g_id70_virtual_grafting_comparison_metadata.json",
        {
            "display_statistic": "absolute Spearman correlation",
            "signed_statistics_table": str(args.stats_table),
            "error_bars": "95% donor-block bootstrap confidence intervals after absolute-value transformation",
            "significance": "Benjamini-Hochberg adjusted donor-cluster sign-flip q values",
            "bar_labels": "absolute rho centered in each bar",
            "legend_placement": "outside upper right",
        },
    )


if __name__ == "__main__":
    main()
