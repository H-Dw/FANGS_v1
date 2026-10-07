#!/usr/bin/env python3
"""Plot normalized LaG16 FANGS, pTM, ipTM and pLDDT values in one panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SHARED_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_SHARED_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SHARED_TOOLS))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision_common import COLORS, apply_style, save_figure, write_metadata


SAMPLE_DIR = "6lr7B_grafting_connector_candidates_temp07_s50_250901"
SELECTED_IDS = (
    "6lr7B_7nowA",
    "6lr7B_4dkaB",
    "6lr7B_8taoC",
    "6lr7B_3k1kC",
    "6lr7B_1zmyA",
)
AFFINITY = {
    "6lr7B_7nowA": 35.0,
    "6lr7B_4dkaB": 69.0,
    "6lr7B_8taoC": 76.0,
    "6lr7B_3k1kC": 120.0,
    "6lr7B_1zmyA": 150.0,
}
ALIASES = {"6lr7B_4dkaA": "6lr7B_4dkaB"}


def minmax(values: pd.Series, invert: bool = False) -> pd.Series:
    low, high = values.min(), values.max()
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Cannot min-max normalize constant or non-finite values in {values.name}")
    normalized = (values - low) / (high - low)
    return 1.0 - normalized if invert else normalized


def load_fangs(base_path: Path) -> pd.DataFrame:
    path = base_path / SAMPLE_DIR / "extract_distance/filter_best/grafted_raw_embeddings.tsv"
    table = pd.read_csv(path, sep="\t")
    table = table[table["PDB_ID"].isin(SELECTED_IDS)].copy()
    if len(table) != len(SELECTED_IDS):
        missing = sorted(set(SELECTED_IDS) - set(table["PDB_ID"]))
        raise ValueError(f"Missing LaG16 FANGS rows: {missing}")
    return table[["PDB_ID", "grafted_euclidean_all_raw_embeddings"]]


def load_protenix(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    required = {"target", "best_ptm", "best_iptm", "best_plddt"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing Protenix columns: {sorted(missing)}")
    table["PDB_ID"] = table["target"].replace(ALIASES)
    return table[["PDB_ID", "best_ptm", "best_iptm", "best_plddt"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path)
    parser.add_argument("--protenix", type=Path)
    parser.add_argument("--source-table", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_table is not None:
        frame = pd.read_csv(args.source_table, sep="\t")
    else:
        if args.base_path is None or args.protenix is None:
            raise RuntimeError("Provide --source-table or both --base-path and --protenix")
        frame = load_fangs(args.base_path).merge(load_protenix(args.protenix), on="PDB_ID", validate="one_to_one")
        frame["Affinity_nM"] = frame["PDB_ID"].map(AFFINITY)
        frame = frame.sort_values("Affinity_nM").reset_index(drop=True)
        frame["FANGS_normalized"] = minmax(frame["grafted_euclidean_all_raw_embeddings"], invert=True)
        frame["pTM_normalized"] = minmax(frame["best_ptm"])
        frame["ipTM_normalized"] = minmax(frame["best_iptm"])
        frame["pLDDT_normalized"] = minmax(frame["best_plddt"])
    frame.to_csv(args.output_dir / "Fig2g_lag16_normalized_scores_source.tsv", sep="\t", index=False)

    stats = []
    for column, label in [
        ("FANGS_normalized", "Connector"),
        ("pTM_normalized", "pTM"),
        ("ipTM_normalized", "ipTM"),
        ("pLDDT_normalized", "pLDDT"),
    ]:
        rho, p = spearmanr(frame["Affinity_nM"], frame[column])
        stats.append({"metric": label, "rho_with_affinity": rho, "p_value": p, "n": len(frame)})
    pd.DataFrame(stats).to_csv(args.output_dir / "Fig2g_lag16_normalized_scores_stats.tsv", sep="\t", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(4.7, 3.05))
    series = [
        ("Connector", "FANGS_normalized", COLORS["fangs"]),
        ("pTM", "pTM_normalized", COLORS["blue"]),
        ("ipTM", "ipTM_normalized", COLORS["ochre"]),
        ("pLDDT", "pLDDT_normalized", COLORS["lavender"]),
    ]
    for label, column, color in series:
        ax.plot(
            frame["Affinity_nM"],
            frame[column],
            color=color,
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=label,
        )
    ax.set_xlabel("Affinity, K$_D$ (nM)")
    ax.set_ylabel("Normalized score")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(left=0.15, right=0.73, bottom=0.18, top=0.96)
    save_figure(fig, args.output_dir / "Fig2g_lag16_normalized_scores")
    plt.close(fig)
    write_metadata(
        args.output_dir / "Fig2g_lag16_normalized_scores_metadata.json",
        {
            "normalization": "independent min-max across the five grafted constructs",
            "direction": "FANGS was inverted so that all normalized series are higher-is-better",
            "markers": "uniform circles",
            "source_protenix": str(args.protenix) if args.protenix else None,
            "legend_placement": "outside upper right",
        },
    )


if __name__ == "__main__":
    main()
