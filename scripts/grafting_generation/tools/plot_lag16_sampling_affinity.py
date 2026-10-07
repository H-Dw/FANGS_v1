#!/usr/bin/env python3
"""Reproduce and simplify the LaG16 affinity_best_score sampling panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SHARED_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_SHARED_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SHARED_TOOLS))

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from revision_common import COLORS, apply_style, save_figure, write_metadata


SAMPLES = {
    5: "6lr7B_grafting_connector_candidates_temp07_s5_250831",
    10: "6lr7B_grafting_connector_candidates_temp07_s10_250901",
    30: "6lr7B_grafting_connector_candidates_temp07_s30_250901",
    50: "6lr7B_grafting_connector_candidates_temp07_s50_250901",
    80: "6lr7B_grafting_connector_candidates_temp07_s80_250901",
    100: "6lr7B_grafting_connector_candidates_temp07_s100_250831",
    500: "6lr7B_grafting_connector_candidates_temp07_s500_250901",
}
AFFINITY = {
    "6lr7B": 31.0,
    "6lr7B_7nowA": 35.0,
    "6lr7B_4dkaB": 69.0,
    "6lr7B_8taoC": 76.0,
    "6lr7B_3k1kC": 120.0,
    "6lr7B_1zmyA": 150.0,
}
SELECTED = tuple(key for key in AFFINITY if key != "6lr7B")
PALETTE = ["#4E79A7", "#76B7B2", "#59A14F", "#D1A24B", "#D78682", "#A388B5", "#92765E"]


def read_one(base: Path, n_sample: int, folder: str) -> pd.DataFrame:
    path = base / folder / "extract_distance/filter_best/grafted_raw_embeddings.tsv"
    table = pd.read_csv(path, sep="\t")
    table = table[table["PDB_ID"].isin(SELECTED)][["PDB_ID", "grafted_euclidean_all_raw_embeddings"]].copy()
    table["Affinity_nM"] = table["PDB_ID"].map(AFFINITY)
    table["n_sample"] = n_sample
    if len(table) != len(SELECTED):
        raise ValueError(f"{folder} contains {len(table)} of {len(SELECTED)} selected constructs")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--source-table", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.source_table is not None:
        combined = pd.read_csv(args.source_table, sep="\t")
    else:
        if args.base_path is None:
            raise RuntimeError("Provide --base-path or --source-table")
        combined = pd.concat([read_one(args.base_path, n, folder) for n, folder in SAMPLES.items()], ignore_index=True)
    combined.to_csv(args.output_dir / "Fig2f_lag16_sampling_affinity_source.tsv", sep="\t", index=False)
    stats = []
    for n_sample, group in combined.groupby("n_sample"):
        rho, p = spearmanr(group["Affinity_nM"], group["grafted_euclidean_all_raw_embeddings"])
        stats.append({"n_sample": n_sample, "rho": rho, "p_value": p, "n_constructs": len(group)})
    pd.DataFrame(stats).to_csv(args.output_dir / "Fig2f_lag16_sampling_affinity_stats.tsv", sep="\t", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.75))
    for color, (n_sample, group) in zip(PALETTE, combined.groupby("n_sample", sort=True)):
        group = group.sort_values("Affinity_nM")
        ax.plot(
            group["Affinity_nM"],
            group["grafted_euclidean_all_raw_embeddings"],
            color=color,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=f"n = {n_sample}",
        )
    ymin, ymax = ax.get_ylim()
    label_top = ymax - 0.015 * (ymax - ymin)
    for construct, affinity in sorted(AFFINITY.items(), key=lambda item: item[1]):
        ax.axvline(affinity, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.85)
        ax.text(
            affinity + 1.2,
            label_top,
            f"{construct}, {affinity:.0f} nM",
            rotation=90,
            ha="left",
            va="top",
            color=COLORS["gray"],
            fontsize=7,
            clip_on=False,
        )
    ax.set_xlim(27, 164)
    ax.set_xlabel("Affinity, K$_D$ (nM)")
    ax.set_ylabel("Connector representation")
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(left=0.11, right=0.79, bottom=0.18, top=0.96)
    save_figure(fig, args.output_dir / "Fig2f_lag16_sampling_affinity")
    plt.close(fig)
    write_metadata(
        args.output_dir / "Fig2f_lag16_sampling_affinity_metadata.json",
        {
            "sample_sizes": list(SAMPLES),
            "vertical_lines": AFFINITY,
            "line_style": "all affinity references use dark gray dashed lines",
            "legend": "sample size only",
            "legend_placement": "outside upper right",
        },
    )


if __name__ == "__main__":
    main()
