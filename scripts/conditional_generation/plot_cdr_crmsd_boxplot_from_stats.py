#!/usr/bin/env python3
"""Publication-style boxplots for CDR1/2/3 cRMSD comparing two model outputs.

Reads per-sample values from TSV files co-located with the statistics summaries
(default: all_info/all_generation.tsv vs rf3_template/rf3_results.tsv). The
statistics.txt paths are used to locate TSVs and for figure annotations.

Use --split-figures to write two separate figures (one per model) when y-ranges
differ strongly so each panel gets its own y-axis scale.

If only mean/std are available (no TSV), falls back to bar + errorbar from
parsed statistics (see --allow-summary-only).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


CDR_COLS = ["CDR1_cRMSD", "CDR2_cRMSD", "CDR3_cRMSD"]
CDR_LABELS = ["CDR1", "CDR2", "CDR3"]

# Publication-oriented palette (colorblind-friendly blue / orange)
COLOR_A = "#2C73B9"
COLOR_B = "#E67E22"


def _default_tsv_for_stats(stats_path: Path) -> Path | None:
    """Guess TSV next to a statistics file (same directory)."""
    parent = stats_path.resolve().parent
    name = stats_path.name.lower()
    candidates: list[str] = []
    if name == "statistics.txt":
        candidates.extend(["all_generation.tsv", "processed_data.tsv"])
    if "rf3" in str(stats_path) or name.startswith("rf3"):
        candidates.append("rf3_results.tsv")
    candidates.extend(["all_generation.tsv", "rf3_results.tsv", "processed_data.csv"])
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        p = parent / c
        if p.is_file():
            return p
    return None


def _parse_calc_style_statistics(path: Path) -> dict[str, tuple[float, float]]:
    """Parse all_info/statistics.txt (calc_temp_generation style)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, tuple[float, float]] = {}
    current: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.endswith(":") and "_cRMSD" in line:
            current = line.rstrip(":").strip()
            continue
        if current and line.startswith("Mean:"):
            m = re.search(r"Mean:\s*([\d.+-eE]+)", line)
            s = re.search(r"Std:\s*([\d.+-eE]+)", line)
            if m:
                mean = float(m.group(1))
                std = float(s.group(1)) if s else 0.0
                out[current] = (mean, std)
            current = None
    return out


def _parse_rf3_style_statistics(path: Path) -> dict[str, tuple[float, float]]:
    """Parse rf3_statistics.txt (table with Metric / Mean / Std rows)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("-") or "Metric" in line:
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0] in CDR_COLS:
            try:
                mean = float(parts[1])
                std = float(parts[2])
                out[parts[0]] = (mean, std)
            except (ValueError, IndexError):
                continue
    return out


def load_summary_from_stats(path: Path) -> dict[str, tuple[float, float]]:
    """Mean/std per CDR column from a statistics file."""
    a = _parse_calc_style_statistics(path)
    if a:
        return {k: v for k, v in a.items() if k in CDR_COLS}
    b = _parse_rf3_style_statistics(path)
    if b:
        return b
    raise ValueError(f"Could not parse CDR cRMSD mean/std from: {path}")


def load_long_from_tsv(tsv_path: Path, model_label: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    missing = [c for c in CDR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{tsv_path}: missing columns {missing}")
    rows: list[dict] = []
    for _, r in df.iterrows():
        for col, lab in zip(CDR_COLS, CDR_LABELS):
            v = r[col]
            if pd.isna(v):
                continue
            rows.append({"CDR": lab, "cRMSD": float(v), "model": model_label})
    return pd.DataFrame(rows)


def set_publication_rcParams(font_scale: float = 1.0) -> None:
    """Font sizes tuned for ~single-column or two-column figures."""
    base = 10.0 * font_scale
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "font.size": base,
            "axes.labelsize": base + 1,
            "axes.titlesize": base + 1,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "legend.fontsize": base - 0.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.0,
        }
    )


def plot_boxplot_long(
    data: pd.DataFrame,
    label_a: str,
    label_b: str,
    out_path: Path,
    figsize: tuple[float, float],
) -> None:
    """Grouped boxplots; left/bottom spines meet at the corner (no trim/offset gap)."""
    order_cdr = CDR_LABELS
    hue_order = [label_a, label_b]
    palette = {label_a: COLOR_A, label_b: COLOR_B}

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=data,
        x="CDR",
        y="cRMSD",
        hue="model",
        order=order_cdr,
        hue_order=hue_order,
        palette=palette,
        width=0.65,
        linewidth=0.9,
        fliersize=2,
        ax=ax,
        dodge=True,
    )
    ax.set_xlabel("")
    ax.set_xticks(range(len(order_cdr)))
    ax.set_xticklabels(order_cdr)
    ax.set_ylabel("cRMSD (Å)", fontweight="normal")

    # Only left + bottom spines; full length so x/y axes join (avoid trim/offset disconnect).
    sns.despine(ax=ax, top=True, right=True, trim=False, offset=0)
    ax.spines["left"].set_linewidth(mpl.rcParams["axes.linewidth"])
    ax.spines["bottom"].set_linewidth(mpl.rcParams["axes.linewidth"])

    old_leg = ax.get_legend()
    if old_leg is not None:
        old_leg.remove()

    # Legend entries: colors match box fills; text comes from CLI (--label-a / --label-b).
    patch_handles = [
        Patch(facecolor=COLOR_A, edgecolor="0.25", linewidth=0.85),
        Patch(facecolor=COLOR_B, edgecolor="0.25", linewidth=0.85),
    ]
    ax.legend(
        patch_handles,
        [label_a, label_b],
        title=None,
        frameon=True,
        fancybox=False,
        edgecolor="0.35",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.14, right=0.72, top=0.96, bottom=0.14)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_boxplot_single_model(
    data_one: pd.DataFrame,
    label: str,
    color: str,
    out_path: Path,
    figsize: tuple[float, float],
) -> None:
    """One model per figure: y-axis auto-scales to that model (avoids shared-axis compression)."""
    order_cdr = CDR_LABELS
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=data_one,
        x="CDR",
        y="cRMSD",
        order=order_cdr,
        color=color,
        width=0.55,
        linewidth=0.9,
        fliersize=2,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_xticks(range(len(order_cdr)))
    ax.set_xticklabels(order_cdr)
    ax.set_ylabel("cRMSD (Å)", fontweight="normal")
    ax.set_title(label, fontsize=mpl.rcParams["axes.titlesize"], pad=6)

    sns.despine(ax=ax, top=True, right=True, trim=False, offset=0)
    ax.spines["left"].set_linewidth(mpl.rcParams["axes.linewidth"])
    ax.spines["bottom"].set_linewidth(mpl.rcParams["axes.linewidth"])

    fig.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.14)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_bar_summary(
    summary_a: dict[str, tuple[float, float]],
    summary_b: dict[str, tuple[float, float]],
    label_a: str,
    label_b: str,
    out_path: Path,
    figsize: tuple[float, float],
) -> None:
    x = np.arange(len(CDR_LABELS))
    w = 0.35
    means_a = [summary_a[c][0] for c in CDR_COLS]
    stds_a = [summary_a[c][1] for c in CDR_COLS]
    means_b = [summary_b[c][0] for c in CDR_COLS]
    stds_b = [summary_b[c][1] for c in CDR_COLS]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w / 2, means_a, w, yerr=stds_a, label=label_a, color=COLOR_A, capsize=2, ecolor="0.25")
    ax.bar(x + w / 2, means_b, w, yerr=stds_b, label=label_b, color=COLOR_B, capsize=2, ecolor="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(CDR_LABELS)
    ax.set_ylabel("cRMSD (Å)")
    ax.legend(frameon=False)
    sns.despine(trim=True)
    plt.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CDR1/2/3 cRMSD boxplot comparing two runs (blue vs orange)."
    )
    parser.add_argument(
        "--stats-a",
        type=Path,
        default=Path(
            "final_version/data/ESM3-Template_validation/all_info/statistics.txt"
        ),
        help="First statistics.txt (locates TSV in same folder by default).",
    )
    parser.add_argument(
        "--stats-b",
        type=Path,
        default=Path(
            "final_version/data/ESM3-Template_validation/rf3_template/rf3_statistics.txt"
        ),
        help="Second statistics file (e.g. rf3_statistics.txt).",
    )
    parser.add_argument("--tsv-a", type=Path, default=None, help="Override TSV for model A.")
    parser.add_argument("--tsv-b", type=Path, default=None, help="Override TSV for model B.")
    parser.add_argument(
        "--label-a",
        type=str,
        default="ESM3 (all_info)",
        help="Legend label for model A (blue).",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default="RF3",
        help="Legend label for model B (orange).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output base path without extension (writes .pdf and .png).",
    )
    parser.add_argument(
        "--allow-summary-only",
        action="store_true",
        help="If TSVs are missing, plot mean±std bars from statistics files instead.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(3.6, 3.0),
        metavar=("W", "H"),
        help="Figure size in inches (width height), e.g. 3.6 3.0 for single column.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiply base font sizes (1.0 = ~10 pt body).",
    )
    parser.add_argument(
        "--split-figures",
        action="store_true",
        help=(
            "Write two separate figures (model A and model B) with independent y-scales; "
            "useful when magnitudes differ strongly. Filenames get _A / _B suffixes."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    stats_a = args.stats_a if args.stats_a.is_absolute() else repo_root / args.stats_a
    stats_b = args.stats_b if args.stats_b.is_absolute() else repo_root / args.stats_b

    for p in (stats_a, stats_b):
        if not p.is_file():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    set_publication_rcParams(args.font_scale)

    tsv_a = args.tsv_a
    if tsv_a is None:
        tsv_a = _default_tsv_for_stats(stats_a)
    else:
        tsv_a = tsv_a if tsv_a.is_absolute() else repo_root / tsv_a

    tsv_b = args.tsv_b
    if tsv_b is None:
        tsv_b = _default_tsv_for_stats(stats_b)
    else:
        tsv_b = tsv_b if tsv_b.is_absolute() else repo_root / tsv_b

    out_base = args.output
    if out_base is None:
        out_dir = stats_a.parent.parent / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_base = out_dir / "cdr_crmsd_boxplot_two_models"
    else:
        out_base = out_base if out_base.is_absolute() else repo_root / out_base
        out_base.parent.mkdir(parents=True, exist_ok=True)

    use_box = tsv_a and tsv_b and tsv_a.is_file() and tsv_b.is_file()

    if use_box:
        try:
            dfa = load_long_from_tsv(tsv_a, args.label_a)
            dfb = load_long_from_tsv(tsv_b, args.label_b)
        except ValueError as e:
            print(f"ERROR loading TSV: {e}", file=sys.stderr)
            sys.exit(1)
        data = pd.concat([dfa, dfb], ignore_index=True)
        if args.split_figures:
            out_a = out_base.with_name(f"{out_base.name}_A")
            out_b = out_base.with_name(f"{out_base.name}_B")
            plot_boxplot_single_model(
                dfa,
                args.label_a,
                COLOR_A,
                out_a,
                figsize=(args.figsize[0], args.figsize[1]),
            )
            plot_boxplot_single_model(
                dfb,
                args.label_b,
                COLOR_B,
                out_b,
                figsize=(args.figsize[0], args.figsize[1]),
            )
            print("Split boxplots saved (independent y-axes):")
            print(f"  A ({args.label_a}): {out_a.with_suffix('.pdf')}")
            print(f"  B ({args.label_b}): {out_b.with_suffix('.pdf')}")
        else:
            plot_boxplot_long(
                data,
                args.label_a,
                args.label_b,
                out_base,
                figsize=(args.figsize[0], args.figsize[1]),
            )
            print(f"Boxplot saved: {out_base.with_suffix('.pdf')} , {out_base.with_suffix('.png')}")
        n_a = len(pd.read_csv(tsv_a, sep="\t"))
        print(f"  Data: {tsv_a.name} ({n_a} rows), {tsv_b.name}")
    elif args.allow_summary_only:
        try:
            sa = load_summary_from_stats(stats_a)
            sb = load_summary_from_stats(stats_b)
        except ValueError as e:
            print(f"ERROR parsing statistics: {e}", file=sys.stderr)
            sys.exit(1)
        for c in CDR_COLS:
            if c not in sa or c not in sb:
                print(f"ERROR: missing {c} in one of the statistics files.", file=sys.stderr)
                sys.exit(1)
        out_bar = out_base.with_name(out_base.name.replace("boxplot", "bar_mean_std"))
        plot_bar_summary(sa, sb, args.label_a, args.label_b, out_bar, tuple(args.figsize))
        print(
            f"Mean±std bar chart saved (no per-sample TSV): "
            f"{out_bar.with_suffix('.pdf')}"
        )
    else:
        print(
            "ERROR: Could not find TSV files for box plots.\n"
            f"  Model A: tried {tsv_a}\n"
            f"  Model B: tried {tsv_b}\n"
            "Pass --tsv-a / --tsv-b or use --allow-summary-only for mean±std bars.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
