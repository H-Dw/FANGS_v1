#!/usr/bin/env python3
"""Plot RF3 vs Protenix best iptm / ptm / plddt vs experimental affinity (nM)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Fallback if --affinity is missing: same as affinity_lag16.tsv + chain alias for prediction IDs.
DEFAULT_AFFINITY_ALIASES: dict[str, float] = {
    "6lr7B_4dkaA": 69.0,  # predictions use chain A; experimental row may be 4dkaB
}

# Colorblind-friendly model colors (Okabe–Ito inspired)
COLOR_RF3 = "#0072B2"
COLOR_PROTENIX = "#D55E00"

# Distinct markers per construct (protein / complex), not per model
TARGET_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "8")


def apply_publication_style() -> None:
    """Font sizes and line widths typical for single-column journal figures (~85 mm wide)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 8,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.linewidth": 1.4,
            "lines.markersize": 6,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Nimbus Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def read_best_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    need = {"target", "best_iptm", "best_ptm", "best_plddt"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def load_affinity_map(path: Path) -> dict[str, float]:
    aff = pd.read_csv(path, sep="\t")
    id_col = "PDB_ID" if "PDB_ID" in aff.columns else "target"
    val_col = "Affinity_nM" if "Affinity_nM" in aff.columns else "Affinity"
    m = dict(zip(aff[id_col].astype(str), aff[val_col].astype(float)))
    m.update(DEFAULT_AFFINITY_ALIASES)
    return m


def attach_affinity(df: pd.DataFrame, affinity_map: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["Affinity_nM"] = out["target"].map(affinity_map)
    missing = out[out["Affinity_nM"].isna()]["target"].tolist()
    if missing:
        raise ValueError(f"No affinity entry for target(s): {missing}")
    return out


def target_marker_map(targets: list[str]) -> dict[str, str]:
    """Stable one marker per target (alphabetical by target for reproducibility)."""
    uniq = sorted(set(targets))
    if len(uniq) > len(TARGET_MARKERS):
        raise ValueError(f"Need more marker types: {len(uniq)} targets")
    return {t: TARGET_MARKERS[i] for i, t in enumerate(uniq)}


def _plot_model_series(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    targets: list[str],
    tmap: dict[str, str],
    color: str,
) -> None:
    """Line colored by model; markers encode target (construct)."""
    ax.plot(x, y, color=color, linestyle="-", linewidth=1.5, zorder=1, alpha=0.95)
    for xi, yi, tgt in zip(x, y, targets):
        ax.scatter(
            xi,
            yi,
            marker=tmap[tgt],
            s=42,
            facecolors=color,
            edgecolors="0.15",
            linewidths=0.45,
            zorder=3,
            clip_on=False,
        )


def plot_metrics(
    rf3: pd.DataFrame,
    protenix: pd.DataFrame,
    out_path: Path | None,
) -> None:
    apply_publication_style()

    rf3 = rf3.sort_values("Affinity_nM").reset_index(drop=True)
    protenix = protenix.sort_values("Affinity_nM").reset_index(drop=True)

    x = rf3["Affinity_nM"].to_numpy(dtype=float)
    if not np.array_equal(protenix["Affinity_nM"].to_numpy(), x):
        raise ValueError("RF3 and Protenix rows must align on affinity order; check targets.")

    targets = rf3["target"].astype(str).tolist()
    tmap = target_marker_map(targets)

    metrics = [
        ("best_iptm", "ipTM"),
        ("best_ptm", "pTM"),
        ("best_plddt", "pLDDT"),
    ]

    # ~180 mm width for 3 panels → ~2.35 in per panel at 7.1 in total
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (col, ylab) in zip(axes, metrics):
        _plot_model_series(ax, x, rf3[col].to_numpy(), targets, tmap, COLOR_RF3)
        _plot_model_series(ax, x, protenix[col].to_numpy(), targets, tmap, COLOR_PROTENIX)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Affinity (nM)")
        ax.minorticks_on()
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ----- Combined legend: model (color) + construct (marker); targets by ascending affinity -----
    leg_model = [
        Line2D([0], [0], color=COLOR_RF3, lw=2.4, label="RosettaFold3"),
        Line2D([0], [0], color=COLOR_PROTENIX, lw=2.4, label="Protenix"),
    ]
    ordered_targets = rf3["target"].astype(str).tolist()
    leg_targets = [
        Line2D(
            [0],
            [0],
            marker=tmap[t],
            color="0.2",
            linestyle="None",
            markersize=7,
            markerfacecolor="0.85",
            markeredgecolor="0.2",
            markeredgewidth=0.45,
            label=t,
        )
        for t in ordered_targets
    ]

    n_items = len(leg_model) + len(leg_targets)
    ncol = 4 if n_items >= 8 else min(n_items, 4)
    fig.legend(
        handles=leg_model + leg_targets,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="0.4",
        framealpha=1.0,
        columnspacing=1.15,
        handletextpad=0.65,
        borderpad=0.4,
    )

    fig.align_ylabels(axes)
    plt.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.33, wspace=0.5)

    if out_path:
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rf3",
        type=Path,
        default=Path(__file__).resolve().parent / "rf3_best.tsv",
    )
    p.add_argument(
        "--protenix",
        type=Path,
        default=Path(__file__).resolve().parent / "protenix_best.tsv",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (.png / .pdf / .svg); default: interactive window",
    )
    p.add_argument(
        "--affinity",
        type=Path,
        default=Path(__file__).resolve().parent / "affinity_lag16.tsv",
        help="TSV with PDB_ID (or target) and Affinity_nM (or Affinity)",
    )
    args = p.parse_args()

    affinity_map = load_affinity_map(args.affinity)
    rf3 = attach_affinity(read_best_tsv(args.rf3), affinity_map)
    protenix = attach_affinity(read_best_tsv(args.protenix), affinity_map)

    t1, t2 = set(rf3["target"]), set(protenix["target"])
    if t1 != t2:
        raise ValueError(f"Target mismatch: RF3 {t1 ^ t2}")

    plot_metrics(rf3, protenix, args.output)


if __name__ == "__main__":
    main()
