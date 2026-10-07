#!/usr/bin/env python3
"""Analyze regional and threshold sensitivity in the id70 grafting benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

_SHARED_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_SHARED_TOOLS) not in sys.path:
    sys.path.insert(0, str(_SHARED_TOOLS))

from revision_common import COLORS, benjamini_hochberg, cluster_spearman


SELECT_COLUMN = "grafted_euclidean_all_raw_embeddings_norm"
ORIGINAL_COLUMN = "original_euclidean_all_raw_embeddings_norm"
LEGACY_COLUMN = "legacy_donor_referenced_absolute_change"
DIRECT_COLUMN = "direct_carrier_graft_all_raw_norm"
KEY_COLUMNS = ["Donator", "FR_ID", "Filename"]
REGIONS = ["connector1", "connector2", "connector3"]
OUTCOMES = ["cRMSD", "pTM"]
STAGES = ["original", "grafted"]
ANCHOR_DEFAULT = [5, 8, 10, 12, 15, 20, 30]


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_spearman(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    if mask.sum() < 3:
        return float("nan")
    if np.unique(left_array[mask]).size < 2 or np.unique(right_array[mask]).size < 2:
        return float("nan")
    value = spearmanr(left_array[mask], right_array[mask])
    return float(getattr(value, "correlation", value[0]))


def tie_plan(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.ones(len(values), dtype=bool)
    if len(values) > 1:
        starts[1:] = sorted_values[1:] != sorted_values[:-1]
    sorted_groups = np.cumsum(starts) - 1
    groups = np.empty(len(values), dtype=np.int32)
    groups[order] = sorted_groups
    group_starts = np.flatnonzero(starts)
    return order, group_starts, groups


def weighted_midranks(
    weights: np.ndarray,
    order: np.ndarray,
    group_starts: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    ordered_weights = weights[:, order]
    group_weights = np.add.reduceat(ordered_weights, group_starts, axis=1)
    before = np.cumsum(group_weights, axis=1) - group_weights
    midpoints = before + (group_weights + 1.0) / 2.0
    return midpoints[:, groups]


def weighted_correlations(
    left_ranks: np.ndarray,
    right_ranks: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    total = weights.sum(axis=1)
    left_mean = np.divide(
        np.sum(weights * left_ranks, axis=1),
        total,
        out=np.full(len(total), np.nan),
        where=total > 0,
    )
    right_mean = np.divide(
        np.sum(weights * right_ranks, axis=1),
        total,
        out=np.full(len(total), np.nan),
        where=total > 0,
    )
    left_centered = left_ranks - left_mean[:, None]
    right_centered = right_ranks - right_mean[:, None]
    covariance = np.sum(weights * left_centered * right_centered, axis=1)
    left_ss = np.sum(weights * left_centered * left_centered, axis=1)
    right_ss = np.sum(weights * right_centered * right_centered, axis=1)
    denominator = np.sqrt(left_ss * right_ss)
    return np.divide(
        covariance,
        denominator,
        out=np.full(len(total), np.nan),
        where=denominator > 0,
    )


def crossed_two_way_bootstrap(
    frame: pd.DataFrame,
    pairs: list[tuple[str, str, str]],
    n_resamples: int,
    seed: int,
    batch_size: int = 256,
) -> dict[str, dict[str, float | int]]:
    columns = sorted({column for _, left, right in pairs for column in (left, right)})
    data = frame[["Donator", "FR_ID", *columns]].dropna().copy()
    if len(data) < 3 or data["Donator"].nunique() < 2 or data["FR_ID"].nunique() < 2:
        return {
            label: {
                "rho": safe_spearman(data[left], data[right]),
                "two_way_ci_low": float("nan"),
                "two_way_ci_high": float("nan"),
                "two_way_valid_resamples": 0,
            }
            for label, left, right in pairs
        }

    donor_codes, donors = pd.factorize(data["Donator"], sort=True)
    carrier_codes, carriers = pd.factorize(data["FR_ID"], sort=True)
    n_donors = len(donors)
    n_carriers = len(carriers)
    arrays = {column: data[column].to_numpy(dtype=float) for column in columns}
    group_cache = {column: tie_plan(values) for column, values in arrays.items()}
    boot = {label: np.full(n_resamples, np.nan, dtype=float) for label, _, _ in pairs}
    rng = np.random.default_rng(seed)

    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        current = stop - start
        donor_counts = rng.multinomial(
            n_donors,
            np.full(n_donors, 1.0 / n_donors),
            size=current,
        )
        carrier_counts = rng.multinomial(
            n_carriers,
            np.full(n_carriers, 1.0 / n_carriers),
            size=current,
        )
        weights = donor_counts[:, donor_codes] * carrier_counts[:, carrier_codes]
        ranks = {}
        for column in columns:
            order, group_starts, groups = group_cache[column]
            ranks[column] = weighted_midranks(weights, order, group_starts, groups)
        for label, left, right in pairs:
            boot[label][start:stop] = weighted_correlations(
                ranks[left],
                ranks[right],
                weights,
            )

    results = {}
    for label, left, right in pairs:
        valid = boot[label][np.isfinite(boot[label])]
        if len(valid):
            low, high = np.quantile(valid, [0.025, 0.975])
            abs_low, abs_high = np.quantile(np.abs(valid), [0.025, 0.975])
        else:
            low, high = float("nan"), float("nan")
            abs_low, abs_high = float("nan"), float("nan")
        rho = safe_spearman(data[left], data[right])
        results[label] = {
            "rho": rho,
            "abs_rho": abs(rho),
            "two_way_ci_low": float(low),
            "two_way_ci_high": float(high),
            "two_way_abs_ci_low": float(abs_low),
            "two_way_abs_ci_high": float(abs_high),
            "two_way_valid_resamples": int(len(valid)),
            "n": int(len(data)),
            "n_donors": int(n_donors),
            "n_carriers": int(n_carriers),
        }
    return results


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_selected_designs(
    merged_path: Path,
    official_selected_path: Path,
) -> pd.DataFrame:
    columns = [
        "Donator",
        "FR_ID",
        "Filename",
        "cRMSD",
        "pTM",
        ORIGINAL_COLUMN,
        SELECT_COLUMN,
        "change_euclidean_all_raw_embeddings",
    ]
    for region in REGIONS:
        for stage in STAGES:
            columns.append(f"{stage}_euclidean_{region}_raw_embeddings_norm")
    all_samples = pd.read_csv(merged_path, sep="\t", usecols=columns)
    minima = all_samples.groupby(["Donator", "FR_ID"])[SELECT_COLUMN].transform("min")
    candidates = all_samples[all_samples[SELECT_COLUMN] == minima].copy()
    selected = (
        candidates.sort_values(["Donator", "FR_ID", SELECT_COLUMN, "Filename"], kind="mergesort")
        .groupby(["Donator", "FR_ID"], sort=True, as_index=False)
        .head(1)
        .sort_values(["Donator", "FR_ID"])
        .reset_index(drop=True)
    )
    official = pd.read_csv(official_selected_path, sep="\t", usecols=KEY_COLUMNS)
    official = official.sort_values(["Donator", "FR_ID"]).reset_index(drop=True)
    if len(selected) != len(official):
        raise RuntimeError(
            f"Selected design count differs from official table: {len(selected)} versus {len(official)}"
        )
    comparison = selected[KEY_COLUMNS].astype(str).reset_index(drop=True).eq(
        official[KEY_COLUMNS].astype(str).reset_index(drop=True)
    )
    if not comparison.all().all():
        mismatch = int((~comparison.all(axis=1)).sum())
        raise RuntimeError(f"Selected Filename differs from official table for {mismatch} cells")
    selected[LEGACY_COLUMN] = np.abs(selected[ORIGINAL_COLUMN] - selected[SELECT_COLUMN])
    maximum_error = float(
        np.max(
            np.abs(
                selected[LEGACY_COLUMN]
                - selected["change_euclidean_all_raw_embeddings"]
            )
        )
    )
    if maximum_error > 1e-10:
        raise RuntimeError(f"Legacy change definition audit failed with max error {maximum_error}")
    return selected


def merge_direct(selected: pd.DataFrame, direct_path: Path) -> pd.DataFrame:
    direct = pd.read_csv(direct_path, sep="\t")
    if direct.duplicated(["Donator", "FR_ID"]).any():
        raise RuntimeError("Direct metric table contains duplicate donor-carrier cells")
    merged = selected.merge(
        direct,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if merged[DIRECT_COLUMN].isna().any():
        raise RuntimeError(f"Direct metrics are missing for {int(merged[DIRECT_COLUMN].isna().sum())} cells")
    return merged


def one_way_pair_stats(
    frame: pd.DataFrame,
    left: str,
    right: str,
    n_resamples: int,
    seed: int,
) -> dict:
    if len(frame) < 3 or frame[left].nunique() < 2 or frame[right].nunique() < 2:
        return {
            "rho": safe_spearman(frame[left], frame[right]),
            "abs_rho": abs(safe_spearman(frame[left], frame[right])),
            "donor_ci_low": float("nan"),
            "donor_ci_high": float("nan"),
            "donor_p": float("nan"),
            "carrier_ci_low": float("nan"),
            "carrier_ci_high": float("nan"),
            "carrier_p": float("nan"),
        }
    donor = cluster_spearman(
        frame,
        left,
        right,
        "Donator",
        n_resamples=n_resamples,
        seed=seed,
    )
    carrier = cluster_spearman(
        frame,
        left,
        right,
        "FR_ID",
        n_resamples=n_resamples,
        seed=seed + 1,
    )
    return {
        "rho": donor["rho"],
        "abs_rho": donor["abs_rho"],
        "donor_ci_low": donor["ci_low"],
        "donor_ci_high": donor["ci_high"],
        "donor_abs_ci_low": donor["abs_ci_low"],
        "donor_abs_ci_high": donor["abs_ci_high"],
        "donor_p": donor["cluster_randomization_p"],
        "carrier_ci_low": carrier["ci_low"],
        "carrier_ci_high": carrier["ci_high"],
        "carrier_abs_ci_low": carrier["abs_ci_low"],
        "carrier_abs_ci_high": carrier["abs_ci_high"],
        "carrier_p": carrier["cluster_randomization_p"],
    }


def connector_region_statistics(
    selected: pd.DataFrame,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    pairs = []
    definitions = []
    for stage in STAGES:
        for region in REGIONS:
            left = f"{stage}_euclidean_{region}_raw_embeddings_norm"
            for outcome in OUTCOMES:
                label = f"{stage}|{region}|{outcome}"
                pairs.append((label, left, outcome))
                definitions.append((label, stage, region, left, outcome))
    crossed = crossed_two_way_bootstrap(selected, pairs, n_resamples, seed)
    rows = []
    for index, (label, stage, region, left, outcome) in enumerate(definitions):
        one_way = one_way_pair_stats(
            selected,
            left,
            outcome,
            n_resamples,
            seed + 100 + index * 3,
        )
        rows.append(
            {
                "stage": stage,
                "region": region,
                "representation": "RAW",
                "outcome": outcome,
                **one_way,
                **{key: value for key, value in crossed[label].items() if key != "rho"},
                "n_cells": int(len(selected)),
                "n_donors": int(selected["Donator"].nunique()),
                "n_carriers": int(selected["FR_ID"].nunique()),
                "n_resamples": int(n_resamples),
            }
        )
    result = pd.DataFrame(rows)
    valid = result["donor_p"].notna()
    result.loc[valid, "q_value"] = benjamini_hochberg(result.loc[valid, "donor_p"].tolist())
    return result


def threshold_point_curves(
    selected: pd.DataFrame,
    thresholds: list[int],
    manuscript_only: bool = False,
) -> pd.DataFrame:
    metrics = {
        "direct carrier-graft representation": DIRECT_COLUMN,
        "legacy donor-referenced absolute change": LEGACY_COLUMN,
    }
    rows = []
    total = len(selected)
    for metric_label, metric_column in metrics.items():
        if manuscript_only and metric_column != LEGACY_COLUMN:
            continue
        for threshold in thresholds:
            subset = selected[selected[metric_column] <= threshold]
            for outcome in OUTCOMES:
                for stage, representation_column in [
                    ("original", ORIGINAL_COLUMN),
                    ("grafted", SELECT_COLUMN),
                ]:
                    if manuscript_only and stage != "grafted":
                        continue
                    rows.append(
                        {
                            "metric": metric_label,
                            "metric_column": metric_column,
                            "threshold": threshold,
                            "outcome": outcome,
                            "stage": stage,
                            "rho": safe_spearman(subset[representation_column], subset[outcome]),
                            "n_cells": int(len(subset)),
                            "retained_design_percent": 100.0 * len(subset) / total,
                            "n_donors": int(subset["Donator"].nunique()),
                            "n_carriers": int(subset["FR_ID"].nunique()),
                        }
                    )
    return pd.DataFrame(rows)


def threshold_anchor_statistics(
    selected: pd.DataFrame,
    anchors: list[int],
    n_resamples: int,
    seed: int,
    manuscript_only: bool = False,
) -> pd.DataFrame:
    metrics = {
        "direct carrier-graft representation": DIRECT_COLUMN,
        "legacy donor-referenced absolute change": LEGACY_COLUMN,
    }
    rows = []
    offset = 0
    for metric_label, metric_column in metrics.items():
        if manuscript_only and metric_column != LEGACY_COLUMN:
            # Retain the seed offsets of the original two-metric analysis.
            offset += 1000 * (len(anchors) + 1)
            continue
        for threshold in [None, *anchors]:
            subset = selected if threshold is None else selected[selected[metric_column] <= threshold]
            pairs = []
            definitions = []
            for outcome in OUTCOMES:
                for stage, representation_column in [
                    ("original", ORIGINAL_COLUMN),
                    ("grafted", SELECT_COLUMN),
                ]:
                    label = f"{metric_label}|{threshold}|{outcome}|{stage}"
                    pairs.append((label, representation_column, outcome))
                    definitions.append((label, outcome, stage, representation_column))
            crossed = crossed_two_way_bootstrap(
                subset,
                [pair for pair, definition in zip(pairs, definitions) if definition[2] == "grafted"] if manuscript_only else pairs,
                n_resamples,
                seed + offset,
            )
            offset += 1000
            for index, (label, outcome, stage, representation_column) in enumerate(definitions):
                if manuscript_only and stage != "grafted":
                    continue
                if manuscript_only:
                    rho = safe_spearman(subset[representation_column], subset[outcome])
                    one_way = {"rho": rho, "abs_rho": abs(rho)}
                else:
                    one_way = one_way_pair_stats(subset, representation_column, outcome, n_resamples, seed + offset + index * 3)
                rows.append(
                    {
                        "metric": metric_label,
                        "metric_column": metric_column,
                        "threshold": "unfiltered" if threshold is None else int(threshold),
                        "outcome": outcome,
                        "stage": stage,
                        **one_way,
                        **{key: value for key, value in crossed[label].items() if key != "rho"},
                        "n_cells": int(len(subset)),
                        "retained_design_percent": 100.0 * len(subset) / len(selected),
                        "n_donors": int(subset["Donator"].nunique()),
                        "n_carriers": int(subset["FR_ID"].nunique()),
                        "n_resamples": int(n_resamples),
                    }
                )
    return pd.DataFrame(rows)


def cumulative_within_donor_subset(
    frame: pd.DataFrame,
    percentage: int,
) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("Donator", sort=True):
        ordered = group.sort_values([ORIGINAL_COLUMN, "FR_ID", "Filename"], kind="mergesort")
        count = max(1, int(math.ceil(percentage * len(ordered) / 100.0)))
        pieces.append(ordered.head(count))
    if not pieces:
        return frame.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def within_donor_statistics(
    selected: pd.DataFrame,
    percentages: list[int],
    n_resamples: int,
    seed: int,
    manuscript_only: bool = False,
) -> pd.DataFrame:
    metrics = {
        "direct carrier-graft representation": DIRECT_COLUMN,
        "legacy donor-referenced absolute change": LEGACY_COLUMN,
    }
    rows = []
    offset = 0
    for metric_label, metric_column in metrics.items():
        if manuscript_only and metric_column != LEGACY_COLUMN:
            offset += 1000 * len(percentages)
            continue
        filtered = selected[selected[metric_column] <= 10].copy()
        filtered["negative_pTM"] = -filtered["pTM"]
        for percentage in percentages:
            subset = cumulative_within_donor_subset(filtered, percentage)
            pairs = [
                ("cRMSD", SELECT_COLUMN, "cRMSD"),
                ("negative_pTM", SELECT_COLUMN, "negative_pTM"),
            ]
            crossed = crossed_two_way_bootstrap(
                subset,
                pairs,
                n_resamples,
                seed + offset,
            )
            offset += 1000
            for index, (label, left, right) in enumerate(pairs):
                if manuscript_only:
                    rho = safe_spearman(subset[left], subset[right])
                    one_way = {"rho": rho, "abs_rho": abs(rho)}
                else:
                    one_way = one_way_pair_stats(subset, left, right, n_resamples, seed + offset + index * 3)
                rows.append(
                    {
                        "metric": metric_label,
                        "metric_column": metric_column,
                        "threshold": 10,
                        "within_donor_top_percent": percentage,
                        "outcome": "cRMSD" if label == "cRMSD" else "−pTM",
                        **one_way,
                        **{key: value for key, value in crossed[label].items() if key != "rho"},
                        "n_cells": int(len(subset)),
                        "n_donors": int(subset["Donator"].nunique()),
                        "n_carriers": int(subset["FR_ID"].nunique()),
                        "n_resamples": int(n_resamples),
                    }
                )
    return pd.DataFrame(rows)


def threshold_assessment(
    selected: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    metrics = {
        "direct carrier-graft representation": DIRECT_COLUMN,
        "legacy donor-referenced absolute change": LEGACY_COLUMN,
    }
    selected = selected.copy()
    selected["cRMSD_pass"] = selected["cRMSD"] < 1.5
    selected["pTM_pass"] = selected["pTM"] > 0.8
    selected["joint_structure_pass"] = selected["cRMSD_pass"] & selected["pTM_pass"]
    rows = []
    for metric_label, metric_column in metrics.items():
        if selected["joint_structure_pass"].nunique() == 2:
            auc = float(
                roc_auc_score(
                    selected["joint_structure_pass"].astype(int),
                    -selected[metric_column],
                )
            )
        else:
            auc = float("nan")
        for threshold in [None, 8, 10, 12]:
            subset = selected if threshold is None else selected[selected[metric_column] <= threshold]
            donor_retention = []
            for donor, group in selected.groupby("Donator"):
                retained = subset[subset["Donator"] == donor]
                donor_retention.append(100.0 * len(retained) / len(group))
            anchor_label = "unfiltered" if threshold is None else str(threshold)
            correlation_rows = anchors[
                (anchors["metric"] == metric_label)
                & (anchors["threshold"].astype(str) == anchor_label)
                & (anchors["stage"] == "grafted")
            ]
            by_outcome = correlation_rows.set_index("outcome")
            row = {
                "metric": metric_label,
                "metric_column": metric_column,
                "threshold": anchor_label,
                "n_cells": int(len(subset)),
                "retained_design_percent": 100.0 * len(subset) / len(selected),
                "n_donors": int(subset["Donator"].nunique()),
                "n_carriers": int(subset["FR_ID"].nunique()),
                "cRMSD_pass_percent": 100.0 * float(subset["cRMSD_pass"].mean()) if len(subset) else float("nan"),
                "pTM_pass_percent": 100.0 * float(subset["pTM_pass"].mean()) if len(subset) else float("nan"),
                "joint_structure_pass_percent": 100.0 * float(subset["joint_structure_pass"].mean()) if len(subset) else float("nan"),
                "quality_auc_using_negative_metric": auc,
                "donor_retention_min_percent": float(np.min(donor_retention)),
                "donor_retention_median_percent": float(np.median(donor_retention)),
                "donor_retention_max_percent": float(np.max(donor_retention)),
            }
            for outcome in OUTCOMES:
                if outcome in by_outcome.index:
                    stats = by_outcome.loc[outcome]
                    row[f"grafted_{outcome}_rho"] = stats["rho"]
                    row[f"grafted_{outcome}_donor_ci_low"] = stats["donor_ci_low"]
                    row[f"grafted_{outcome}_donor_ci_high"] = stats["donor_ci_high"]
                    row[f"grafted_{outcome}_carrier_ci_low"] = stats["carrier_ci_low"]
                    row[f"grafted_{outcome}_carrier_ci_high"] = stats["carrier_ci_high"]
                    row[f"grafted_{outcome}_two_way_ci_low"] = stats["two_way_ci_low"]
                    row[f"grafted_{outcome}_two_way_ci_high"] = stats["two_way_ci_high"]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_connector_regions(
    stats: pd.DataFrame,
    stage: str,
    output_dir: Path,
) -> None:
    apply_style()
    subset = stats[stats["stage"] == stage].copy()
    fig, ax = plt.subplots(figsize=(3.50, 2.65))
    x = np.arange(len(REGIONS))
    width = 0.32
    outcome_colors = {"cRMSD": COLORS["fangs"], "pTM": COLORS["blue"]}
    for offset, outcome in [(-width / 2, "cRMSD"), (width / 2, "pTM")]:
        rows = subset[subset["outcome"] == outcome].set_index("region").loc[REGIONS]
        values = rows["abs_rho"].to_numpy(dtype=float)
        lower = rows["two_way_abs_ci_low"].to_numpy(dtype=float)
        upper = rows["two_way_abs_ci_high"].to_numpy(dtype=float)
        lower = np.minimum(lower, values)
        upper = np.maximum(upper, values)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=outcome_colors[outcome],
            label=outcome,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="none",
            ecolor=COLORS["gray"],
            elinewidth=0.8,
            capsize=2,
        )
        for bar, (_, row) in zip(bars, rows.iterrows()):
            x_text = bar.get_x() + bar.get_width() / 2.0
            ax.text(
                x_text,
                max(float(row["abs_rho"]) * 0.50, 0.055),
                f"{float(row['abs_rho']):.3f}",
                ha="center",
                va="center",
                rotation=90,
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )
            q_value = float(row["q_value"])
            significance = "***" if q_value < 0.001 else "**" if q_value < 0.01 else "*" if q_value < 0.05 else "ns"
            ax.text(
                x_text,
                min(float(row["two_way_abs_ci_high"]) + 0.055, 1.035),
                significance,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, ["Connector 1", "Connector 2", "Connector 3"])
    ax.set_ylabel("|Spearman ρ|")
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=1)
    fig.subplots_adjust(left=0.18, right=0.75, bottom=0.16, top=0.96)
    filename = (
        "FigS2b_original_connector_regions_spearman"
        if stage == "original"
        else "FigS2c_grafted_connector_regions_spearman"
    )
    save_figure(fig, output_dir / filename)
    plt.close(fig)
    write_json(
        output_dir / f"{filename}_metadata.json",
        {
            "stage": stage,
            "representation": "Connector",
            "n_cells": int(subset["n_cells"].max()),
            "n_donors": int(subset["n_donors"].max()),
            "n_carriers": int(subset["n_carriers"].max()),
            "error_bars": "95% crossed donor-carrier two-way bootstrap CI",
            "display_statistic": "absolute Spearman correlation",
            "signed_statistics_table": "id70_connector_region_spearman.tsv",
            "significance": "Benjamini-Hochberg adjusted donor-label permutation q values",
            "legend_placement": "outside upper right",
            "n_resamples": int(subset["n_resamples"].max()),
        },
    )


def plot_threshold_outcomes(
    curves: pd.DataFrame,
    anchors: pd.DataFrame,
    output_dir: Path,
    table_dir: Path | None = None,
) -> None:
    apply_style()
    metric = "legacy donor-referenced absolute change"
    current = curves[(curves["metric"] == metric) & (curves["stage"] == "grafted")].copy()
    fig, ax = plt.subplots(figsize=(5.25, 2.85))
    for outcome, color, label in [
        ("cRMSD", COLORS["fangs"], "cRMSD"),
        ("pTM", COLORS["blue"], "pTM"),
    ]:
        line = current[current["outcome"] == outcome].sort_values("threshold")
        ax.plot(line["threshold"], line["rho"].abs(), color=color, label=label)
        marker_rows = line[line["threshold"].astype(int) % 5 == 0]
        ax.plot(
            marker_rows["threshold"],
            marker_rows["rho"].abs(),
            linestyle="none",
            marker="o",
            markersize=3.2,
            color=color,
        )
        ci_rows = anchors[
            (anchors["metric"] == metric)
            & (anchors["outcome"] == outcome)
            & (anchors["stage"] == "grafted")
            & (anchors["threshold"] != "unfiltered")
        ].copy()
        ci_rows["threshold_numeric"] = pd.to_numeric(ci_rows["threshold"])
        ci_rows = ci_rows.sort_values("threshold_numeric")
        values = ci_rows["abs_rho"].to_numpy(dtype=float)
        lower = np.minimum(ci_rows["two_way_abs_ci_low"].to_numpy(dtype=float), values)
        upper = np.maximum(ci_rows["two_way_abs_ci_high"].to_numpy(dtype=float), values)
        ax.errorbar(
            ci_rows["threshold_numeric"],
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="none",
            ecolor=color,
            alpha=0.65,
            elinewidth=0.7,
            capsize=1.5,
        )
    ax.axvline(10, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 30)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xlabel("Representation threshold")
    ax.set_ylabel("|Spearman ρ|")
    ax.set_title("Donor-referenced representation shift")
    percent_ax = ax.twinx()
    percent = current[current["outcome"] == "cRMSD"].sort_values("threshold")
    percent_ax.plot(
        percent["threshold"],
        percent["retained_design_percent"],
        color=COLORS["ochre"],
        linestyle=":",
        linewidth=1.3,
        label="Retained designs",
    )
    percent_markers = percent[percent["threshold"].astype(int) % 5 == 0]
    percent_ax.plot(
        percent_markers["threshold"],
        percent_markers["retained_design_percent"],
        color=COLORS["ochre"],
        linestyle="none",
        marker="o",
        markersize=3.0,
    )
    percent_ax.set_ylim(0, 105)
    percent_ax.set_ylabel("Retained donor–carrier designs (%)", color=COLORS["ochre"])
    percent_ax.tick_params(axis="y", colors=COLORS["ochre"])
    percent_ax.spines["right"].set_visible(True)
    percent_ax.spines["right"].set_color(COLORS["ochre"])
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = percent_ax.get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        ncol=1,
        bbox_to_anchor=(0.82, 0.92),
    )
    fig.subplots_adjust(left=0.12, right=0.62, bottom=0.20, top=0.86)
    filename = "Fig1h_id70_delta_connector_threshold_scan"
    save_figure(fig, output_dir / filename)
    plt.close(fig)
    current.to_csv(output_dir / f"{filename}_source.tsv", sep="\t", index=False)
    anchors[
        (anchors["metric"] == metric)
        & (anchors["stage"] == "grafted")
        & (anchors["outcome"].isin(["cRMSD", "pTM"]))
    ].to_csv(output_dir / f"{filename}_anchor_statistics.tsv", sep="\t", index=False)
    write_json(
        output_dir / f"{filename}_metadata.json",
        {
            "outcomes": ["cRMSD", "pTM"],
            "displayed_metric": "legacy donor-referenced absolute change",
            "displayed_stage": "grafted",
            "display_statistic": "absolute Spearman correlation",
            "pTM_direction_note": "pTM is inversely associated with structural deterioration",
            "signed_statistics_table": str((output_dir if table_dir is None else table_dir) / "id70_threshold_curves.tsv"),
            "threshold_grid": "0 to 30 inclusive in steps of 1",
            "circle_marker_step": 5,
            "anchor_error_bars": "95% crossed donor-carrier two-way bootstrap CI",
            "retention_unit": "selected donor-carrier designs",
            "excluded_from_plot": [
                "direct carrier-graft representation",
                "original donor-carrier correlations",
            ],
            "legend_placement": "outside upper right",
        },
    )


def plot_within_donor(stats: pd.DataFrame, output_dir: Path, table_dir: Path | None = None) -> None:
    apply_style()
    metric = "legacy donor-referenced absolute change"
    current = stats[stats["metric"] == metric].copy()
    fig, ax = plt.subplots(figsize=(4.40, 2.75))
    for outcome, color, label in [
        ("cRMSD", COLORS["fangs"], "cRMSD"),
        ("−pTM", COLORS["blue"], "pTM"),
    ]:
        rows = current[current["outcome"] == outcome].sort_values("within_donor_top_percent")
        x = rows["within_donor_top_percent"].to_numpy(dtype=float)
        values = rows["abs_rho"].to_numpy(dtype=float)
        lower = np.minimum(rows["two_way_abs_ci_low"].to_numpy(dtype=float), values)
        upper = np.maximum(rows["two_way_abs_ci_high"].to_numpy(dtype=float), values)
        ax.plot(x, values, color=color, marker="o", label=label)
        ax.errorbar(
            x,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="none",
            ecolor=color,
            alpha=0.65,
            elinewidth=0.7,
            capsize=1.5,
        )
    ax.set_title("Donor-referenced ΔD ≤ 10")
    ax.set_xlabel("Cumulative donor-relative neighborhood (%)")
    ax.set_xticks(np.arange(5, 55, 5))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("|Spearman ρ|")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.75, 0.91), ncol=1)
    fig.subplots_adjust(left=0.14, right=0.70, bottom=0.20, top=0.86)
    filename = "Fig1i_id70_donor_cumulative_neighborhood"
    save_figure(fig, output_dir / filename)
    plt.close(fig)
    current.to_csv(output_dir / f"{filename}_source.tsv", sep="\t", index=False)
    write_json(
        output_dir / f"{filename}_metadata.json",
        {
            "ranking_metric": "Original donor-carrier connector representation",
            "subsets": "Nested cumulative top 5% to top 50% within each donor",
            "selection_count": "max(1, ceil(p × n_donor))",
            "outcome_transform": "pTM was negated to align the structural deterioration direction",
            "error_bars": "95% crossed donor-carrier two-way bootstrap CI",
            "display_statistic": "absolute Spearman correlation",
            "signed_statistics_table": str((output_dir if table_dir is None else table_dir) / "id70_within_donor_cumulative_stability.tsv"),
            "legend_placement": "outside upper right",
            "displayed_metric": "legacy donor-referenced absolute change",
            "excluded_from_plot": "direct carrier-graft representation",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-generations", type=Path)
    parser.add_argument("--selected-designs", type=Path)
    parser.add_argument("--direct-metrics", type=Path)
    parser.add_argument(
        "--selected-complete-table",
        type=Path,
        help="Use an audited one-row-per-design table and skip raw generation-table selection",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, help="Directory for figure exports; defaults to --output-dir")
    parser.add_argument("--threshold-min", type=int, default=0)
    parser.add_argument("--threshold-max", type=int, default=30)
    parser.add_argument("--threshold-step", type=int, default=1)
    parser.add_argument("--anchors", type=str, default=",".join(map(str, ANCHOR_DEFAULT)))
    parser.add_argument("--within-donor-percentages", type=str, default="5,10,15,20,25,30,35,40,45,50")
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--manuscript-only", action="store_true", help="Analyze only the donor-referenced Fig. 1h/i data")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir if args.figure_dir is None else args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    anchors = parse_int_list(args.anchors)
    percentages = parse_int_list(args.within_donor_percentages)
    thresholds = list(range(args.threshold_min, args.threshold_max + 1, args.threshold_step))
    if args.manuscript_only and args.selected_complete_table is None:
        parser.error("--manuscript-only requires --selected-complete-table")

    if args.selected_complete_table is not None:
        selected = pd.read_csv(args.selected_complete_table, sep="\t")
        required = {
            *KEY_COLUMNS,
            "cRMSD",
            "pTM",
            ORIGINAL_COLUMN,
            SELECT_COLUMN,
            LEGACY_COLUMN,
        }
        if not args.manuscript_only:
            required.update({DIRECT_COLUMN, *{f"{stage}_euclidean_{region}_raw_embeddings_norm" for stage in STAGES for region in REGIONS}})
        missing = sorted(required.difference(selected.columns))
        if missing:
            raise RuntimeError(f"Selected complete table lacks columns: {missing}")
        if selected.duplicated(["Donator", "FR_ID"]).any():
            raise RuntimeError("Selected complete table contains duplicate donor-carrier cells")
        if args.manuscript_only:
            numeric_columns = ["cRMSD", "pTM", ORIGINAL_COLUMN, SELECT_COLUMN, LEGACY_COLUMN]
            if not np.isfinite(selected[numeric_columns].to_numpy(dtype=float)).all():
                raise RuntimeError("Manuscript input contains nonfinite measurements")
            if not np.allclose(selected[LEGACY_COLUMN], np.abs(selected[ORIGINAL_COLUMN] - selected[SELECT_COLUMN]), atol=1e-10, rtol=0):
                raise RuntimeError("Donor-referenced shift differs from the manuscript definition")
            selected = selected[[*KEY_COLUMNS, *numeric_columns]].copy()
        selected = selected.sort_values(["Donator", "FR_ID", "Filename"]).reset_index(drop=True)
    else:
        missing_arguments = [
            name
            for name, value in [
                ("--merged-generations", args.merged_generations),
                ("--selected-designs", args.selected_designs),
                ("--direct-metrics", args.direct_metrics),
            ]
            if value is None
        ]
        if missing_arguments:
            raise RuntimeError(f"Missing required inputs: {', '.join(missing_arguments)}")
        selected = load_selected_designs(args.merged_generations, args.selected_designs)
        selected = merge_direct(selected, args.direct_metrics)
    if len(selected) != 33856:
        raise RuntimeError(f"Expected 33856 selected design cells, observed {len(selected)}")
    if selected["Donator"].nunique() != 185 or selected["FR_ID"].nunique() != 184:
        raise RuntimeError("Unexpected donor or carrier coverage in selected designs")

    if not args.manuscript_only:
        connector_stats = connector_region_statistics(selected, args.n_resamples, args.seed)
        connector_stats.to_csv(args.output_dir / "id70_connector_region_spearman.tsv", sep="\t", index=False)
    curves = threshold_point_curves(selected, thresholds, manuscript_only=args.manuscript_only)
    curves.to_csv(args.output_dir / "id70_threshold_curves.tsv", sep="\t", index=False)
    anchor_stats = threshold_anchor_statistics(
        selected,
        anchors,
        args.n_resamples,
        args.seed + 100000,
        manuscript_only=args.manuscript_only,
    )
    anchor_stats.to_csv(
        args.output_dir / "id70_threshold_anchor_statistics.tsv",
        sep="\t",
        index=False,
    )
    within = within_donor_statistics(
        selected,
        percentages,
        args.n_resamples,
        args.seed + 200000,
        manuscript_only=args.manuscript_only,
    )
    within.to_csv(
        args.output_dir / "id70_within_donor_cumulative_stability.tsv",
        sep="\t",
        index=False,
    )
    if not args.manuscript_only:
        assessment = threshold_assessment(selected, anchor_stats)
        assessment.to_csv(args.output_dir / "id70_threshold_10_assessment.tsv", sep="\t", index=False)
    selected.to_csv(
        args.output_dir / ("id70_selected_manuscript_designs.tsv" if args.manuscript_only else "id70_selected_complete_rows_with_direct_metrics.tsv"),
        sep="\t",
        index=False,
    )

    if not args.manuscript_only:
        plot_connector_regions(connector_stats, "original", figure_dir)
        plot_connector_regions(connector_stats, "grafted", figure_dir)
    plot_threshold_outcomes(curves, anchor_stats, figure_dir, table_dir=args.output_dir)
    plot_within_donor(within, figure_dir, table_dir=args.output_dir)

    metadata = {
        "figure_contract": {
            "core_conclusion": "Regional RAW connector representations track graft-associated structural variation. Donor-referenced representation shift at 10 is an operational deformation-risk component of the unified Quality filter, not a universal direct-deformation boundary.",
            "archetype": "quantitative grid",
            "backend": "Python",
            "target_output": "Figure 1h and 1i panels",
            "hero_evidence": "Connector-specific signed Spearman correlations",
            "robustness_evidence": "Direct diagnostic and donor-referenced shift curves with donor-relative cumulative sensitivity",
            "reviewer_risk": "Crossed donor-carrier dependence and the distinction between a donor-referenced proxy and direct displacement",
        },
        "merged_generations_sha256": sha256_file(args.merged_generations) if args.merged_generations else None,
        "selected_designs_sha256": sha256_file(args.selected_designs) if args.selected_designs else None,
        "direct_metrics_sha256": sha256_file(args.direct_metrics) if args.direct_metrics else None,
        "selected_complete_table_sha256": sha256_file(args.selected_complete_table) if args.selected_complete_table else None,
        "selection_rule": "Minimum whole-connector donor-graft RAW representation within each realized donor-carrier cell",
        "design_cells": int(len(selected)),
        "donors": int(selected["Donator"].nunique()),
        "carriers": int(selected["FR_ID"].nunique()),
        "n_resamples": int(args.n_resamples),
        "seed": int(args.seed),
        "threshold_grid": thresholds,
        "threshold_anchors": anchors,
        "within_donor_percentages": percentages,
        "quality_filter_proxy_definition": "Absolute difference between donor-carrier and donor-graft RAW connector representations",
        "quality_filter_proxy_cutoff": "ΔD ≤ 10",
        "direct_definition": "Carrier-graft RAW vector Euclidean difference divided by sqrt(12)",
    }
    if args.manuscript_only:
        metadata.pop("figure_contract")
        metadata.pop("direct_definition")
        metadata["manuscript_panels"] = ["Fig. 1h", "Fig. 1i"]
    write_json(args.output_dir / "id70_virtual_grafting_sensitivity_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if not args.manuscript_only:
        print(assessment.to_string(index=False))


if __name__ == "__main__":
    main()
