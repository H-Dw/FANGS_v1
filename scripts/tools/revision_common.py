#!/usr/bin/env python3
"""Shared statistics and plotting utilities for the 2026-08-31 FANGS revision."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import rankdata, spearmanr
except ModuleNotFoundError:
    rankdata = None
    spearmanr = None


SEED = 20260831
COLORS = {
    "fangs": "#5B8E7D",
    "fangs_light": "#C7DEC9",
    "blue": "#6C8EBF",
    "ochre": "#C49A6C",
    "lavender": "#9A86B8",
    "rose": "#C9837A",
    "teal": "#6FA6A1",
    "gray": "#4A4A4A",
    "light_gray": "#D7D7D7",
}


def require_scipy() -> None:
    if rankdata is None or spearmanr is None:
        raise RuntimeError("SciPy is required for statistical analysis but not for plotting-only commands")


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def benjamini_hochberg(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    result = np.empty_like(q)
    result[order] = q
    return result


def cluster_spearman(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    n_resamples: int = 10000,
    seed: int = SEED,
) -> dict[str, float | int]:
    require_scipy()
    data = frame[[x_col, y_col, cluster_col]].dropna().copy()
    rho, raw_p = spearmanr(data[x_col], data[y_col])
    cluster_codes, clusters = pd.factorize(data[cluster_col], sort=True)
    n_clusters = len(clusters)
    rng = np.random.default_rng(seed)

    x_rank = rankdata(data[x_col].to_numpy(dtype=float))
    y_rank = rankdata(data[y_col].to_numpy(dtype=float))

    cluster_stats = np.zeros((n_clusters, 6), dtype=float)
    for cluster_index in range(n_clusters):
        mask = cluster_codes == cluster_index
        x_values = x_rank[mask]
        y_values = y_rank[mask]
        cluster_stats[cluster_index] = (
            mask.sum(),
            x_values.sum(),
            y_values.sum(),
            np.square(x_values).sum(),
            np.square(y_values).sum(),
            np.multiply(x_values, y_values).sum(),
        )

    boot = np.empty(n_resamples, dtype=float)
    batch_size = 512
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        draws = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        counts = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=n_clusters),
            1,
            draws,
        )
        totals = counts @ cluster_stats
        n, sx, sy, sxx, syy, sxy = totals.T
        numerator = sxy - sx * sy / n
        denominator = np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
        boot[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.full(stop - start, np.nan, dtype=float),
            where=denominator > 0,
        )
    valid_boot = boot[np.isfinite(boot)]
    low, high = np.quantile(valid_boot, [0.025, 0.975])
    abs_low, abs_high = np.quantile(np.abs(valid_boot), [0.025, 0.975])

    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    cross_products = np.bincount(
        cluster_codes,
        weights=x_centered * y_centered,
        minlength=n_clusters,
    )
    x_norm = np.sqrt(np.sum(x_centered * x_centered))
    y_norm = np.sqrt(np.sum(y_centered * y_centered))
    denominator = x_norm * y_norm
    null_abs = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(stop - start, n_clusters),
        )
        null_abs[start:stop] = np.abs(signs @ cross_products / denominator)
    cluster_p = float((1 + np.sum(null_abs >= abs(float(rho)))) / (n_resamples + 1))

    return {
        "rho": float(rho),
        "abs_rho": abs(float(rho)),
        "ci_low": float(low),
        "ci_high": float(high),
        "abs_ci_low": float(abs_low),
        "abs_ci_high": float(abs_high),
        "raw_spearman_p": float(raw_p),
        "cluster_randomization_p": cluster_p,
        "n": int(len(data)),
        "n_clusters": int(n_clusters),
        "n_resamples": int(n_resamples),
    }


def cluster_spearman_difference(
    frame: pd.DataFrame,
    reference_col: str,
    comparator_col: str,
    outcome_col: str,
    cluster_col: str,
    n_resamples: int = 10000,
    seed: int = SEED,
) -> dict[str, float | int]:
    require_scipy()
    data = frame[[reference_col, comparator_col, outcome_col, cluster_col]].dropna().copy()
    codes, clusters = pd.factorize(data[cluster_col], sort=True)
    n_clusters = len(clusters)
    reference_rank = rankdata(data[reference_col].to_numpy(dtype=float))
    comparator_rank = rankdata(data[comparator_col].to_numpy(dtype=float))
    outcome_rank = rankdata(data[outcome_col].to_numpy(dtype=float))

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.corrcoef(left, right)[0, 1])

    observed_reference = correlation(reference_rank, outcome_rank)
    observed_comparator = correlation(comparator_rank, outcome_rank)
    observed_difference = abs(observed_reference) - abs(observed_comparator)

    def summaries(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        result = np.zeros((n_clusters, 6), dtype=float)
        for cluster_index in range(n_clusters):
            mask = codes == cluster_index
            x_values = left[mask]
            y_values = right[mask]
            result[cluster_index] = (
                mask.sum(),
                x_values.sum(),
                y_values.sum(),
                np.square(x_values).sum(),
                np.square(y_values).sum(),
                np.multiply(x_values, y_values).sum(),
            )
        return result

    reference_stats = summaries(reference_rank, outcome_rank)
    comparator_stats = summaries(comparator_rank, outcome_rank)
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    batch_size = 512
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        draws = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        counts = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=n_clusters),
            1,
            draws,
        )

        def weighted_correlations(stats: np.ndarray) -> np.ndarray:
            n, sx, sy, sxx, syy, sxy = (counts @ stats).T
            numerator = sxy - sx * sy / n
            denominator = np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
            return np.divide(
                numerator,
                denominator,
                out=np.full(stop - start, np.nan, dtype=float),
                where=denominator > 0,
            )

        differences[start:stop] = np.abs(weighted_correlations(reference_stats)) - np.abs(
            weighted_correlations(comparator_stats)
        )
    valid = differences[np.isfinite(differences)]
    low, high = np.quantile(valid, [0.025, 0.975])
    two_sided_p = float(
        2.0
        * min(
            (1 + np.sum(valid <= 0.0)) / (len(valid) + 1),
            (1 + np.sum(valid >= 0.0)) / (len(valid) + 1),
        )
    )
    return {
        "reference_rho": observed_reference,
        "comparator_rho": observed_comparator,
        "reference_abs_rho": abs(observed_reference),
        "comparator_abs_rho": abs(observed_comparator),
        "abs_rho_difference": observed_difference,
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_two_sided_p": min(two_sided_p, 1.0),
        "n": int(len(data)),
        "n_clusters": int(n_clusters),
        "n_resamples": int(n_resamples),
    }


def significance_label(q: float) -> str:
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def place_legend_upper_right(
    ax: plt.Axes,
    *,
    ncol: int = 1,
    x_anchor: float = 1.02,
    y_anchor: float = 1.0,
) -> None:
    """Place a plot legend outside the upper-right edge."""
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(x_anchor, y_anchor),
        borderaxespad=0.0,
        ncol=ncol,
    )


def annotate_abs_spearman_bar(
    ax: plt.Axes,
    bar: mpl.patches.Patch,
    abs_rho: float,
    abs_ci_high: float,
    q_value: float,
) -> None:
    """Label a magnitude bar with rho inside and significance above."""
    x = bar.get_x() + bar.get_width() / 2.0
    compact = abs_rho < 0.16
    y_center = abs_rho * 0.50
    ax.text(
        x,
        y_center,
        f"{abs_rho:.3f}",
        ha="center",
        va="center",
        rotation=0 if compact else 90,
        fontsize=6.3 if compact else 7.5,
        color="white" if abs_rho >= 0.18 else COLORS["gray"],
        fontweight="bold",
    )
    ax.text(
        x,
        min(abs_ci_high + 0.055, 1.035),
        significance_label(float(q_value)),
        ha="center",
        va="bottom",
        fontsize=8,
    )


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
