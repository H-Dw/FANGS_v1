#!/usr/bin/env python3
"""Summarize the id70 CDR-by-FR benchmark and create Fig. 1g."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from scipy.stats import spearmanr

_SHARED_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_SHARED_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SHARED_TOOLS))

from revision_common import COLORS, annotate_abs_spearman_bar, apply_style, benjamini_hochberg, cluster_spearman, cluster_spearman_difference, place_legend_upper_right, save_figure, write_metadata


SELECT_COLUMN = "grafted_euclidean_all_raw_embeddings_norm"
ALIGNER = PairwiseAligner()
ALIGNER.mode = "global"
ALIGNER.match_score = 2.0
ALIGNER.mismatch_score = -1.0
ALIGNER.open_gap_score = -2.0
ALIGNER.extend_gap_score = -2.0


def sequence_identity(left: str, right: str) -> float:
    alignment = ALIGNER.align(left, right)[0]
    coordinates = alignment.coordinates
    matches = 0
    alignment_length = 0
    for segment in range(coordinates.shape[1] - 1):
        left_start, left_stop = coordinates[0, segment : segment + 2]
        right_start, right_stop = coordinates[1, segment : segment + 2]
        left_span = int(left_stop - left_start)
        right_span = int(right_stop - right_start)
        alignment_length += max(left_span, right_span)
        if left_span and right_span:
            matches += sum(
                left[int(left_start) + offset] == right[int(right_start) + offset]
                for offset in range(min(left_span, right_span))
            )
    return matches / alignment_length if alignment_length else float("nan")


def run_usalign(task: tuple[str, str, Path, Path, Path]) -> dict:
    left_id, right_id, left_path, right_path, usalign = task
    command = [
        str(usalign), str(left_path), str(right_path),
        "-mol", "prot", "-ter", "2", "-outfmt", "2", "-fast",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [line for line in completed.stdout.splitlines() if line and not line.startswith("#")]
    if not lines:
        raise RuntimeError(f"USalign returned no tabular line for {left_id} and {right_id}")
    fields = lines[-1].split("\t")
    if len(fields) < 11:
        raise RuntimeError(f"Unexpected USalign output for {left_id} and {right_id}: {lines[-1]}")
    return {
        "native_1": left_id,
        "native_2": right_id,
        "tm_score_1": float(fields[2]),
        "tm_score_2": float(fields[3]),
        "global_structural_similarity": (float(fields[2]) + float(fields[3])) / 2.0,
        "global_rmsd": float(fields[4]),
        "usalign_identity_aligned": float(fields[7]),
        "length_1": int(fields[8]),
        "length_2": int(fields[9]),
        "aligned_length": int(fields[10]),
    }


def compute_controls(cdr_info: Path, structure_dir: Path, usalign: Path, output: Path, workers: int) -> pd.DataFrame:
    if output.exists():
        return pd.read_csv(output, sep="\t")
    info = pd.read_csv(cdr_info, sep="\t").drop_duplicates("PDBChain")
    sequences = dict(zip(info["PDBChain"], info["Sequence"]))
    identifiers = sorted(sequences)
    tasks = [
        (left, right, structure_dir / f"{left}.pdb", structure_dir / f"{right}.pdb", usalign)
        for left, right in combinations(identifiers, 2)
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_usalign, tasks))
    for row in rows:
        row["global_sequence_identity"] = sequence_identity(
            sequences[row["native_1"]], sequences[row["native_2"]]
        )
    controls = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    controls.to_csv(output, sep="\t", index=False)
    return controls


def directed_controls(controls: pd.DataFrame) -> pd.DataFrame:
    forward = controls.rename(columns={"native_1": "Donator", "native_2": "FR_ID"})
    reverse = controls.rename(columns={"native_2": "Donator", "native_1": "FR_ID"})
    return pd.concat([forward, reverse], ignore_index=True)


def make_main_stats(selected: pd.DataFrame, output_dir: Path, n_resamples: int, seed: int, figure_dir: Path | None = None) -> pd.DataFrame:
    metrics = [
        (SELECT_COLUMN, "Connector"),
        ("global_structural_similarity", "Global structure"),
        ("global_sequence_identity", "Global sequence"),
    ]
    outcomes = [("cRMSD", "cRMSD"), ("pTM", "pTM")]
    rows = []
    index = 0
    for metric, metric_label in metrics:
        for outcome, outcome_label in outcomes:
            stats = cluster_spearman(
                selected,
                metric,
                outcome,
                "Donator",
                n_resamples=n_resamples,
                seed=seed + index,
            )
            rows.append(
                {
                    "metric": metric,
                    "metric_label": metric_label,
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    **stats,
                }
            )
            index += 1
    stats_df = pd.DataFrame(rows)
    stats_df["q_value"] = benjamini_hochberg(stats_df["cluster_randomization_p"].tolist())
    stats_df.to_csv(output_dir / "id70_main_correlation_stats.tsv", sep="\t", index=False)
    figure_dir = output_dir if figure_dir is None else figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(figure_dir / "Fig1g_id70_virtual_grafting_comparison_source.tsv", sep="\t", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(4.7, 2.85))
    metric_labels = [value[1] for value in metrics]
    xpos = np.arange(len(metric_labels))
    width = 0.34
    colors = {"cRMSD": COLORS["fangs"], "pTM": COLORS["blue"]}
    for offset, (_, outcome_label) in zip((-width / 2, width / 2), outcomes):
        subset = stats_df[stats_df["outcome_label"] == outcome_label].set_index("metric_label").loc[metric_labels]
        yerr = np.vstack(
            [subset["abs_rho"] - subset["abs_ci_low"], subset["abs_ci_high"] - subset["abs_rho"]]
        )
        bars = ax.bar(
            xpos + offset,
            subset["abs_rho"],
            width=width,
            color=colors[outcome_label],
            label=outcome_label,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.errorbar(xpos + offset, subset["abs_rho"], yerr=yerr, fmt="none", ecolor=COLORS["gray"], capsize=2, linewidth=0.8)
        for bar, (_, row) in zip(bars, subset.iterrows()):
            annotate_abs_spearman_bar(ax, bar, row["abs_rho"], row["abs_ci_high"], row["q_value"])
    ax.set_ylabel("|Spearman ρ|")
    ax.set_xticks(xpos, metric_labels)
    ax.set_ylim(0.0, 1.08)
    place_legend_upper_right(ax, ncol=1)
    fig.subplots_adjust(left=0.14, right=0.76, bottom=0.20, top=0.94)
    save_figure(fig, figure_dir / "Fig1g_id70_virtual_grafting_comparison")
    plt.close(fig)
    write_metadata(
        figure_dir / "Fig1g_id70_virtual_grafting_comparison_metadata.json",
        {
            "display_statistic": "absolute Spearman correlation",
            "signed_statistics_table": str(output_dir / "id70_main_correlation_stats.tsv"),
            "error_bars": "95% donor-block bootstrap confidence intervals after absolute-value transformation",
            "significance": "Benjamini-Hochberg adjusted donor-label permutation q values",
            "bar_labels": "absolute rho centered in each bar",
            "legend_placement": "outside upper right",
        },
    )
    return stats_df


def supplemental_stats(all_samples: pd.DataFrame, selected: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metrics = [
        ("original_euclidean_all_raw_embeddings_norm", "RAW original"),
        ("grafted_euclidean_all_pre_q_embeddings_norm", "pre-VQ grafted"),
        ("grafted_euclidean_all_ca_distance_norm", "Cα grafted"),
    ]
    rows = []
    for column, label in metrics:
        for outcome in ("cRMSD", "pTM"):
            rho, p = spearmanr(selected[column], selected[outcome], nan_policy="omit")
            rows.append({"aggregation": "selected best sample", "metric": label, "outcome": outcome, "rho": rho, "p_value": p, "n": len(selected)})
            rho_all, p_all = spearmanr(all_samples[column], all_samples[outcome], nan_policy="omit")
            rows.append({"aggregation": "all generated samples", "metric": label, "outcome": outcome, "rho": rho_all, "p_value": p_all, "n": len(all_samples)})
    result = pd.DataFrame(rows)
    result["abs_rho"] = result["rho"].abs()
    result.to_csv(output_dir / "id70_supplementary_partial_representation_stats.tsv", sep="\t", index=False)
    return result


def block_sensitivity_stats(
    selected: pd.DataFrame,
    output_dir: Path,
    n_resamples: int,
    seed: int,
) -> None:
    metrics = [
        (SELECT_COLUMN, "Connector"),
        ("global_structural_similarity", "Global structure"),
        ("global_sequence_identity", "Global sequence"),
    ]
    rows = []
    offset = 0
    for cluster_col in ("Donator", "FR_ID"):
        for metric, label in metrics:
            for outcome in ("cRMSD", "pTM"):
                rows.append(
                    {
                        "resampling_block": cluster_col,
                        "metric": metric,
                        "metric_label": label,
                        "outcome": outcome,
                        **cluster_spearman(
                            selected,
                            metric,
                            outcome,
                            cluster_col,
                            n_resamples=n_resamples,
                            seed=seed + offset,
                        ),
                    }
                )
                offset += 1
    sensitivity = pd.DataFrame(rows)
    sensitivity["q_value"] = benjamini_hochberg(
        sensitivity["cluster_randomization_p"].tolist()
    )
    sensitivity.to_csv(
        output_dir / "id70_donor_carrier_block_sensitivity.tsv",
        sep="\t",
        index=False,
    )

    comparisons = []
    for comparator, label in [
        ("global_structural_similarity", "Global structure"),
        ("global_sequence_identity", "Global sequence"),
    ]:
        for outcome in ("cRMSD", "pTM"):
            comparisons.append(
                {
                    "reference": "Connector",
                    "comparator": label,
                    "outcome": outcome,
                    **cluster_spearman_difference(
                        selected,
                        SELECT_COLUMN,
                        comparator,
                        outcome,
                        "Donator",
                        n_resamples=n_resamples,
                        seed=seed + offset,
                    ),
                }
            )
            offset += 1
    pd.DataFrame(comparisons).to_csv(
        output_dir / "id70_donor_block_correlation_differences.tsv",
        sep="\t",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-generations", type=Path, required=True)
    parser.add_argument("--cdr-info", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--usalign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, help="Directory for native controls and selected designs; defaults to --output-dir")
    parser.add_argument("--figure-dir", type=Path, help="Directory for figure exports; defaults to --output-dir")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--manuscript-only", action="store_true", help="Write the Fig. 1g analysis without exploratory comparisons")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir if args.tables_dir is None else args.tables_dir
    figure_dir = args.output_dir if args.figure_dir is None else args.figure_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    controls = compute_controls(
        args.cdr_info,
        args.structure_dir,
        args.usalign,
        tables_dir / "id70_native_pairwise_global_controls.tsv",
        args.workers,
    )
    columns = [
        "Donator", "FR_ID", "Filename", "cRMSD", "pTM",
        SELECT_COLUMN,
        "original_euclidean_all_raw_embeddings_norm",
    ]
    if not args.manuscript_only:
        columns.extend(["grafted_euclidean_all_pre_q_embeddings_norm", "grafted_euclidean_all_ca_distance_norm"])
    all_samples = pd.read_csv(args.merged_generations, sep="\t", usecols=columns)
    index = all_samples.groupby(["Donator", "FR_ID"], sort=False)[SELECT_COLUMN].idxmin()
    selected = all_samples.loc[index].copy()
    selected = selected.merge(
        directed_controls(controls),
        on=["Donator", "FR_ID"],
        how="left",
        validate="many_to_one",
    )
    if selected[["global_structural_similarity", "global_sequence_identity"]].isna().any().any():
        missing = selected[selected["global_structural_similarity"].isna()][["Donator", "FR_ID"]]
        raise RuntimeError(f"Global controls missing for {len(missing)} selected design cells")
    selected = selected.sort_values(["Donator", "FR_ID"]).reset_index(drop=True)
    selected.to_csv(tables_dir / "id70_selected_best_sample_per_design.tsv", sep="\t", index=False)
    main_stats = make_main_stats(selected, args.output_dir, args.n_resamples, args.seed, figure_dir=figure_dir)
    if not args.manuscript_only:
        supplemental_stats(all_samples, selected, args.output_dir)
        block_sensitivity_stats(selected, args.output_dir, args.n_resamples, args.seed)
    metadata = {
        "sequence_representatives": int(pd.read_csv(args.cdr_info, sep="\t")["PDBChain"].nunique()),
        "planned_directed_cells": 34040,
        "realized_directed_cells": int(len(selected)),
        "generated_samples": int(len(all_samples)),
        "samples_per_realized_cell_min": int(all_samples.groupby(["Donator", "FR_ID"]).size().min()),
        "samples_per_realized_cell_max": int(all_samples.groupby(["Donator", "FR_ID"]).size().max()),
        "selection_column": SELECT_COLUMN,
        "selection_rule": "minimum donor-graft RAW connector representation within each realized donor-carrier cell",
        "global_structure_metric": "mean of the two US-align TM-score normalization directions",
        "global_sequence_metric": "Needleman-Wunsch identity over full alignment columns",
        "bootstrap_cluster": "Donator",
        "n_resamples": args.n_resamples,
        "seed": args.seed,
    }
    write_metadata(args.output_dir / "id70_analysis_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    print(main_stats.to_string(index=False))


if __name__ == "__main__":
    main()
