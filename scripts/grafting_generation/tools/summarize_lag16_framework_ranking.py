#!/usr/bin/env python3
"""Summarize the archived LaG16 filter and native/grafted framework ranks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

SELECTED = {
    "6lr7B_7nowA": "6lr7B_7nowA",
    "6lr7B_4dkaA": "6lr7B_4dkaB",
    "6lr7B_8taoC": "6lr7B_8taoC",
    "6lr7B_3k1kC": "6lr7B_3k1kC",
    "6lr7B_1zmyA": "6lr7B_1zmyA",
}
ORIGINAL = "original_euclidean_all_raw_embeddings"
GRAFTED = "grafted_euclidean_all_raw_embeddings"
CHANGE = "change_euclidean_all_raw_embeddings"


def ranked(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    if frame.PDB_ID.duplicated().any() or not np.isfinite(frame[score]).all():
        raise ValueError("Framework ranking requires unique IDs and finite scores")
    frame = frame.sort_values([score, "PDB_ID"], kind="stable").reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["cumulative_rank_percent"] = 100 * frame["rank"] / len(frame)
    return frame


def draw_pyramid(native: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.5, 6.0))
    cutoffs = [0, .10, .20, .30, .50, 1.0]
    colors = plt.get_cmap("PuBuGn")(np.linspace(.85, .20, 5))
    for start, stop, color in zip(cutoffs[:-1], cutoffs[1:], colors):
        ax.add_patch(Polygon([(-stop, 1-stop), (-start, 1-start),
                             (start, 1-start), (stop, 1-stop)],
                            facecolor=color, edgecolor="black", linewidth=.6))
        ax.text(-1.18, 1-(start+stop)/2, f"{100*start:.0f}–{100*stop:.0f}%",
                ha="right", va="center", fontsize=10)
    highlighted = native[native.PDB_ID.isin(SELECTED)].copy()
    if len(highlighted) != len(SELECTED):
        raise ValueError("The native ranking lacks a selected carrier")
    highlighted["height"] = 1-(highlighted["rank"]-.5)/len(native)
    highlighted = highlighted.sort_values("height", ascending=False)
    positions = highlighted.height.to_numpy().copy()
    for i in range(1, len(positions)):
        positions[i] = min(positions[i], positions[i-1]-.075)
    if positions[-1] < .04:
        positions += .04-positions[-1]
    for (_, row), label_y in zip(highlighted.iterrows(), positions):
        height = float(row.height)
        ax.plot([1-height, 1.18, 1.35], [height, label_y, label_y],
                color="black", linewidth=.7)
        label = SELECTED[row.PDB_ID].split("_", 1)[1]
        ax.text(1.38, label_y, f"{label}  (rank {int(row['rank'])})",
                fontsize=10, va="center")
    ax.set_xlim(-1.75, 2.8)
    ax.set_ylim(-.03, 1.06)
    ax.set_aspect("equal")
    ax.set_title("Native donor–carrier Connector ranking", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(output/f"LaG16_framework_ranking.{ext}",
                    bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_generation": root/"temp_generation/all_info/all_generation.tsv",
        "native": root/"extract_distance/full_avge/original_raw_embeddings.tsv",
        "filtered": root/"extract_distance/filter/grafted_raw_embeddings.tsv",
        "changes": root/"extract_distance/filter/change_raw_embeddings.tsv",
        "best": root/"extract_distance/filter_best/grafted_raw_embeddings.tsv",
    }
    generation = pd.read_csv(paths["all_generation"], sep="\t")
    native = ranked(pd.read_csv(paths["native"], sep="\t"), ORIGINAL)
    filtered = pd.read_csv(paths["filtered"], sep="\t")
    changes = pd.read_csv(paths["changes"], sep="\t")
    keys = ["PDB_ID", "Filename"]
    if filtered.duplicated(keys).any() or changes.duplicated(keys).any():
        raise ValueError("Duplicate generated-sample keys")
    audit = filtered.merge(changes[keys+[CHANGE]], on=keys, validate="one_to_one")
    if len(audit) != len(filtered) or not (audit[CHANGE] <= 10).all():
        raise ValueError("Filtered records do not satisfy the archived ΔConnector cutoff")
    ordered = filtered.sort_values(["PDB_ID", GRAFTED, "Filename"], kind="stable")
    grafted = ordered.drop_duplicates("PDB_ID").copy()
    grafted = grafted.merge(generation[["Filename", "pTM", "cRMSD"]],
                            on="Filename", validate="one_to_one")
    grafted = ranked(grafted, GRAFTED)
    best = pd.read_csv(paths["best"], sep="\t")
    comparison = grafted.merge(best[["PDB_ID", GRAFTED]], on="PDB_ID",
                               suffixes=("", "_archived"), validate="one_to_one")
    if len(comparison) != len(best) or not np.allclose(
        comparison[GRAFTED], comparison[GRAFTED+"_archived"], atol=1e-12, rtol=0
    ):
        raise ValueError("Reconstructed minima differ from the archived best scores")
    native.to_csv(out/"LaG16_ranked_native_frameworks.tsv", sep="\t", index=False)
    grafted.to_csv(out/"LaG16_ranked_grafted_frameworks.tsv", sep="\t", index=False)
    selected = native[native.PDB_ID.isin(SELECTED)][["PDB_ID", ORIGINAL, "rank", "cumulative_rank_percent"]]
    selected = selected.rename(columns={"rank": "native_rank", "cumulative_rank_percent": "native_rank_percent"})
    selected = selected.merge(grafted[["PDB_ID", "Filename", GRAFTED, "rank", "cumulative_rank_percent", "pTM", "cRMSD"]], on="PDB_ID", validate="one_to_one")
    if len(selected) != len(SELECTED):
        raise ValueError("A selected construct is missing from the grafted ranking")
    selected = selected.rename(columns={"PDB_ID": "source_PDB_ID", "rank": "grafted_rank", "cumulative_rank_percent": "grafted_rank_percent"})
    selected.insert(0, "experimental_PDB_ID", selected.source_PDB_ID.map(SELECTED))
    selected.to_csv(out/"LaG16_selected_carrier_ranks.tsv", sep="\t", index=False)
    counts = {
        "generated_samples": len(generation),
        "generated_designs": int(generation.PDB_ID.nunique()),
        "connector_filtered_samples": len(filtered),
        "connector_filtered_percent": 100*len(filtered)/len(generation),
        "connector_filtered_designs": int(filtered.PDB_ID.nunique()),
        "filter": "Absolute donor-referenced RAW Connector distance change ≤ 10",
        "native_ranking_metric": ORIGINAL,
        "grafted_ranking_metric": GRAFTED,
        "grafted_sample_selection": "Minimum whole-connector distance; stable Filename tie break",
        "rank_tie_break": "PDB_ID ascending",
        "source_to_experimental_labels": SELECTED,
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
    }
    (out/"LaG16_filter_counts.json").write_text(json.dumps(counts, indent=2)+"\n")
    draw_pyramid(native, out)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
