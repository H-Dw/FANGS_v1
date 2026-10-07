#!/usr/bin/env python3
"""
Bar plot of insertion scores for CDR fragment combinations.

Each PDB_ID is one bar. Bar length is --values_size. Bar color is a rank
percentile group of --values_color (smaller values = better ranks). Bars are
sorted top-to-bottom by percentile group, then by --values_size ascending
within each group. Pass --top50 to also write a plot that keeps only the
top 50% grafted-rank bars.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.offsetbox import HPacker
from matplotlib.patches import Patch

GROUP_PERCENTILES = [10, 20, 30, 50]
GROUP_LABELS = [
    "Top 0–10%",
    "Top 10–20%",
    "Top 20–30%",
    "Top 30–50%",
    "Rest (>50%)",
]
# Sequential PuBuGn; best ranks (group 0) use the darkest end so they stand out.
GROUP_COLORS = [
    to_hex(plt.cm.PuBuGn(x))
    for x in np.linspace(0.92, 0.32, len(GROUP_LABELS))
]
CDR_FRAG_PAT = re.compile(r"CDR(\d+)-([A-Za-z]+)", flags=re.IGNORECASE)

# Independently configurable font sizes (points). CLI flags override these.
# Set TICK_FONTSIZE to None to auto-fit y-tick labels to bar spacing.
TICK_FONTSIZE = None
AXIS_FONTSIZE = 16.0
LEGEND_FONTSIZE = 16.0
LEGEND_TITLE_FONTSIZE = 16.0
LEGEND_BOTTOM_INCH = 1.0


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Bar plot of CDR insertion combinations. Length = --values_size; "
            "color = rank percentiles of --values_color."
        )
    )
    p.add_argument(
        "--cdr_info",
        required=False,
        default=None,
        help="TSV containing PDBChain and CDR{n} sequences (kept for CLI compatibility).",
    )
    p.add_argument("--values_color", required=True, help="TSV with PDB_ID and color value.")
    p.add_argument("--values_size", required=True, help="TSV with PDB_ID and size value.")
    p.add_argument("--out_dir", required=True, help="Directory to write the bar plot and table.")
    p.add_argument("--cdr_indices", default="1,2,3", help="Comma-separated CDR indices (e.g. 1,2).")
    p.add_argument("--pdbid_col", default="PDB_ID", help="PDB ID column name used in both value tables.")
    p.add_argument(
        "--color_col",
        default="selected_euclidean_average",
        help="Column name for color values in color TSV.",
    )
    p.add_argument(
        "--size_col",
        default="selected_euclidean_average",
        help="Column name for size values in size TSV.",
    )
    p.add_argument(
        "--top50",
        action="store_true",
        help="Also write a bar plot and table that keep only the top 50%% grafted-rank bars.",
    )
    p.add_argument(
        "--tick_fontsize",
        type=float,
        default=TICK_FONTSIZE,
        help="Tick label font size in points. Default: auto-fit to bar spacing.",
    )
    p.add_argument(
        "--axis_fontsize",
        type=float,
        default=AXIS_FONTSIZE,
        help="Axis label font size in points.",
    )
    p.add_argument(
        "--legend_fontsize",
        type=float,
        default=LEGEND_FONTSIZE,
        help="Legend item font size in points.",
    )
    p.add_argument(
        "--legend_title_fontsize",
        type=float,
        default=LEGEND_TITLE_FONTSIZE,
        help="Legend title font size in points.",
    )
    return p.parse_args()


def parse_cdr_indices(cdr_indices_str):
    indices = [int(x.strip()) for x in cdr_indices_str.split(",") if x.strip().isdigit()]
    if not indices:
        raise ValueError(f"No valid CDR indices in '{cdr_indices_str}'.")
    return indices


def extract_cdr_fragments(pdbid, cdr_indices):
    found = {}
    for m in CDR_FRAG_PAT.finditer(str(pdbid)):
        found[int(m.group(1))] = m.group(2)
    frags = []
    for ci in cdr_indices:
        if ci not in found:
            return None
        frags.append(found[ci])
    return frags


def assign_percentile_groups(color_vals):
    color_vals = np.asarray(color_vals, dtype=float)
    thresholds = {p: np.percentile(color_vals, p) for p in GROUP_PERCENTILES}
    group_idx = np.empty(color_vals.shape, dtype=int)
    for i, v in enumerate(color_vals):
        assigned = len(GROUP_COLORS) - 1
        for idx, p in enumerate(GROUP_PERCENTILES):
            if v <= thresholds[p]:
                assigned = idx
                break
        group_idx[i] = assigned
    return group_idx, thresholds


def load_merged_table(args):
    df_color = pd.read_csv(args.values_color, sep="\t", dtype=str)
    df_size = pd.read_csv(args.values_size, sep="\t", dtype=str)

    if args.pdbid_col not in df_color.columns or args.pdbid_col not in df_size.columns:
        raise ValueError(f"PDB ID column '{args.pdbid_col}' must exist in both value tables.")
    if args.color_col not in df_color.columns:
        raise ValueError(f"Color TSV must contain column '{args.color_col}'.")
    if args.size_col not in df_size.columns:
        raise ValueError(f"Size TSV must contain column '{args.size_col}'.")

    color_col_merged = f"color_{args.color_col}"
    size_col_merged = f"size_{args.size_col}"
    df_color_ren = df_color[[args.pdbid_col, args.color_col]].rename(
        columns={args.color_col: color_col_merged}
    )
    df_size_ren = df_size[[args.pdbid_col, args.size_col]].rename(
        columns={args.size_col: size_col_merged}
    )
    merged = pd.merge(df_color_ren, df_size_ren, on=args.pdbid_col, how="inner")
    if merged.empty:
        raise ValueError("Merged table is empty after joining color and size tables on PDB_ID.")

    merged[color_col_merged] = pd.to_numeric(merged[color_col_merged], errors="coerce")
    merged[size_col_merged] = pd.to_numeric(merged[size_col_merged], errors="coerce")
    merged = merged.dropna(subset=[color_col_merged, size_col_merged])
    if merged.empty:
        raise ValueError("No numeric rows remain after converting merged color/size to numeric.")
    return merged, color_col_merged, size_col_merged


def prepare_plot_table(merged, pdbid_col, color_col, size_col, cdr_indices):
    records = []
    skipped = 0
    for _, row in merged.iterrows():
        pdbid = row[pdbid_col]
        frags = extract_cdr_fragments(pdbid, cdr_indices)
        if frags is None:
            skipped += 1
            continue
        records.append(
            {
                pdbid_col: pdbid,
                "tick_label": "-".join(frags),
                "color_value": float(row[color_col]),
                "size_value": float(row[size_col]),
            }
        )
    if not records:
        raise ValueError(
            "No rows contain all requested CDR fragments "
            f"({', '.join(f'CDR{i}' for i in cdr_indices)})."
        )
    plot_df = pd.DataFrame.from_records(records)
    group_idx, thresholds = assign_percentile_groups(plot_df["color_value"].to_numpy())
    plot_df["group_idx"] = group_idx
    plot_df["group_label"] = [GROUP_LABELS[i] for i in group_idx]
    plot_df = plot_df.sort_values(
        by=["group_idx", "size_value", "color_value", "tick_label"],
        kind="mergesort",
    ).reset_index(drop=True)
    return plot_df, skipped, thresholds


def bar_positions(group_idx, gap=0.6):
    positions = []
    pos = 0.0
    prev_group = None
    for g in group_idx:
        if prev_group is not None and g != prev_group:
            pos += gap
        positions.append(pos)
        pos += 1.0
        prev_group = g
    return np.asarray(positions, dtype=float)


def file_prefix(path):
    return os.path.basename(path).split("_")[0]


def choose_tick_fontsize(fig_h, n, min_fs=11.0, max_fs=22.0):
    """Largest font that still fits one horizontal bar of vertical space."""
    ax_height_frac = 0.82
    inches_per_bar = (fig_h * ax_height_frac) / max(n, 1)
    fs = inches_per_bar * 72.0 * 0.70
    return float(np.clip(fs, min_fs, max_fs))


def legend_title_on_same_row(legend, sep=14):
    """Place the legend title to the left of the handles, on one row."""
    packer = HPacker(
        pad=0,
        sep=sep,
        align="center",
        children=[legend._legend_title_box, legend._legend_handle_box],
    )
    packer.set_figure(legend.figure)
    legend._legend_box = packer


def plot_insertion_bars(
    plot_df,
    combo_label,
    size_label,
    color_label,
    out_path,
    tick_fontsize=TICK_FONTSIZE,
    axis_fontsize=AXIS_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE,
    legend_title_fontsize=LEGEND_TITLE_FONTSIZE,
):
    n = len(plot_df)
    ys = bar_positions(plot_df["group_idx"].to_numpy())
    widths = plot_df["size_value"].to_numpy()
    colors = [GROUP_COLORS[i] for i in plot_df["group_idx"].to_numpy()]

    fig_w = 11.0
    fig_h = max(8.0, 0.42 * n + 2.8) + LEGEND_BOTTOM_INCH
    tick_fs = choose_tick_fontsize(fig_h, n) if tick_fontsize is None else float(tick_fontsize)
    axis_fs = float(axis_fontsize)
    legend_fs = float(legend_fontsize)
    legend_title_fs = float(legend_title_fontsize)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)
    ax.barh(
        ys,
        widths,
        height=0.8,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        align="center",
    )

    ax.set_yticks(ys)
    ax.set_yticklabels(plot_df["tick_label"].tolist(), fontsize=tick_fs)
    ax.tick_params(axis="y", pad=6, length=4)
    ax.tick_params(axis="x", labelsize=tick_fs, pad=4, length=4)
    ax.set_ylabel(combo_label, fontsize=axis_fs, labelpad=10)
    ax.set_xlabel(size_label, fontsize=axis_fs, labelpad=8)
    ax.set_ylim(ys.max() + 0.7, ys.min() - 0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    used_groups = sorted(plot_df["group_idx"].unique())
    legend_patches = [Patch(facecolor=GROUP_COLORS[i], edgecolor="black") for i in used_groups]
    legend_labels = [GROUP_LABELS[i] for i in used_groups]
    left_margin, right_margin = 0.22, 0.97
    bottom_frac = LEGEND_BOTTOM_INCH / fig_h
    fig.subplots_adjust(left=left_margin, right=right_margin, top=0.96, bottom=bottom_frac)
    color_legend = fig.legend(
        legend_patches,
        legend_labels,
        title=color_label.capitalize(),
        loc="lower center",
        bbox_to_anchor=((left_margin + right_margin) / 2.0, 0.015),
        bbox_transform=fig.transFigure,
        ncol=len(used_groups),
        title_fontsize=legend_title_fs,
        fontsize=legend_fs,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.6,
        handletextpad=0.5,
        handlelength=1.4,
    )
    color_legend.get_title().set_fontweight("bold")
    legend_title_on_same_row(color_legend)

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar plot: {out_path}")


def write_outputs(plot_df, stem, combo_label, size_label, color_label, fontsizes):
    out_tsv = f"{stem}.tsv"
    plot_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved bar table: {out_tsv}")
    plot_insertion_bars(
        plot_df,
        combo_label=combo_label,
        size_label=size_label.capitalize(),
        color_label=color_label,
        out_path=f"{stem}.png",
        **fontsizes,
    )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.cdr_info is not None and not os.path.exists(args.cdr_info):
        raise FileNotFoundError(f"--cdr_info file not found: {args.cdr_info}")

    cdr_indices = parse_cdr_indices(args.cdr_indices)
    merged, color_col, size_col = load_merged_table(args)
    plot_df, skipped, _thresholds = prepare_plot_table(
        merged, args.pdbid_col, color_col, size_col, cdr_indices
    )
    if skipped:
        print(f"Skipped {skipped} rows missing requested CDR fragments.")

    combo_label = "-".join(f"CDR{i}" for i in cdr_indices)
    color_label = file_prefix(args.values_color)
    size_label = file_prefix(args.values_size)
    fontsizes = {
        "tick_fontsize": args.tick_fontsize,
        "axis_fontsize": args.axis_fontsize,
        "legend_fontsize": args.legend_fontsize,
        "legend_title_fontsize": args.legend_title_fontsize,
    }
    stem = os.path.join(args.out_dir, f"{combo_label}_insertion_barplot")
    write_outputs(plot_df, stem, combo_label, size_label, color_label, fontsizes)

    if args.top50:
        grafted_p50 = float(np.percentile(plot_df["size_value"].to_numpy(), 50))
        top50_df = (
            plot_df[plot_df["size_value"] <= grafted_p50].copy().reset_index(drop=True)
        )
        if top50_df.empty:
            print("No top-50% grafted bars to plot; skipped --top50 output.")
        else:
            print(f"Kept {len(top50_df)}/{len(plot_df)} bars with grafted <= P50 ({grafted_p50:.4f}).")
            top50_stem = f"{stem}_top50"
            write_outputs(top50_df, top50_stem, combo_label, size_label, color_label, fontsizes)

    print("Finished insertion bar plot.")


if __name__ == "__main__":
    main()
