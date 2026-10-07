#!/usr/bin/env python3
"""Draw the Lag16 affinity/yield grouped bar plot as an editable SVG.

Values come from a text table (PDB_ID, Affinity, Yield), not from literals in
this script. SVG text stays as <text> elements so axis labels, tick numbers,
bar annotations, and legend words can be edited in Inkscape or Illustrator.

Adobe Illustrator drops Matplotlib hatch fills: they are SVG <pattern> tiles
whose lines extend outside the tile, and the bars are also clipped to the
axes. Illustrator then paints those bars blank and warns that clipping will
be lost on a round-trip to SVG Tiny. The same export lists every fallback in
Matplotlib's sans-serif stack (Bitstream Vera Sans, Computer Modern Sans
Serif, Geneva, Lucid, Avant Garde, and others). This script draws the yield
hatch as ordinary line paths and names only Arial.

Example:
    python plot_lag16_affinity_yield_barplot.py
    python plot_lag16_affinity_yield_barplot.py \\
        --input Lag16_affinity_yield.txt \\
        --output ../../../figures/Lag16_affinity_yield_barplot.svg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.transforms import IdentityTransform

# Real text, one installed face. A generic "sans-serif" family is expanded
# into Matplotlib's full fallback list, which Illustrator tries to load.
plt.rcParams.update(
    {
        "font.family": "Arial",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    }
)

REQUIRED_COLUMNS = ("PDB_ID", "Affinity", "Yield")
REFERENCE_COLOR = "#808080"
LEGEND_COLOR = "#4C78A8"
# Matplotlib's "///" hatch is a 72 pt tile whose "/" lines satisfy
# x_svg + y_svg = 8 * k. Draw those same lines as paths instead of a pattern.
HATCH_SUM_STEP = 8.0
HATCH_LINEWIDTH = 1.0
YIELD_ALPHA = 0.55


def load_affinity_yield(path: Path) -> pd.DataFrame:
    """Read a whitespace- or tab-separated table with a header row."""
    table = pd.read_csv(path, sep=None, engine="python", comment="#")
    table.columns = [str(column).strip() for column in table.columns]
    missing = [name for name in REQUIRED_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}. Expected {list(REQUIRED_COLUMNS)}."
        )

    frame = table.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["PDB_ID"] = frame["PDB_ID"].astype(str).str.strip()
    frame["Affinity"] = pd.to_numeric(frame["Affinity"], errors="raise")
    frame["Yield"] = pd.to_numeric(frame["Yield"], errors="raise")
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS))
    if frame.empty:
        raise ValueError(f"{path} has no data rows.")
    if frame["PDB_ID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate PDB_ID values.")
    return frame.reset_index(drop=True)


def bar_colors(n_rows: int) -> list:
    """Match the notebook: the reference construct is gray, the rest are PuBuGn."""
    if n_rows < 1:
        raise ValueError("Need at least one row to plot.")
    if n_rows == 1:
        return [REFERENCE_COLOR]
    palette = list(sns.color_palette("PuBuGn", n_colors=n_rows - 1))
    return [REFERENCE_COLOR, *palette]


def slash_segments(rect, figure_height, spacing=HATCH_SUM_STEP, inset=0.0):
    """Return '/' segments inside a y-up rectangle, phased to the SVG hatch.

    *rect* is (x0, y0, x1, y1) in points with the origin at the lower left.
    After the SVG y-flip, these lines are the ones Matplotlib would paint
    with ``x + y = spacing * k``.
    """
    x0, y0, x1, y1 = rect
    x0 += inset
    x1 -= inset
    y0 += inset
    y1 -= inset
    if x1 <= x0 or y1 <= y0:
        return []

    def add_edge(points, px, py):
        if x0 - 1e-6 <= px <= x1 + 1e-6 and y0 - 1e-6 <= py <= y1 + 1e-6:
            points.append((min(max(px, x0), x1), min(max(py, y0), y1)))

    intercepts = (y0 - x0, y0 - x1, y1 - x0, y1 - x1)
    k_min = math.ceil((figure_height - max(intercepts)) / spacing - 1e-9)
    k_max = math.floor((figure_height - min(intercepts)) / spacing + 1e-9)
    segments = []
    for k in range(k_min, k_max + 1):
        intercept = figure_height - spacing * k
        points = []
        add_edge(points, x0, x0 + intercept)
        add_edge(points, x1, x1 + intercept)
        add_edge(points, y0 - intercept, y0)
        add_edge(points, y1 - intercept, y1)
        unique = []
        for point in points:
            if not any(abs(point[0] - kept[0]) < 1e-4 and abs(point[1] - kept[1]) < 1e-4 for kept in unique):
                unique.append(point)
        if len(unique) >= 2:
            unique.sort()
            segments.append((unique[0], unique[-1]))
    return segments


def patch_rect_points(patch) -> tuple[float, float, float, float]:
    """Axis-aligned patch bounds in display points (origin at the lower left)."""
    vertices = patch.get_transform().transform(patch.get_path().vertices)
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def add_yield_hatch(fig, patches) -> None:
    """Paint '/' lines as real strokes so Illustrator does not need a pattern."""
    figure_height = float(fig.get_size_inches()[1] * fig.dpi)
    inset = HATCH_LINEWIDTH / 2.0
    segments = []
    for patch in patches:
        segments.extend(slash_segments(patch_rect_points(patch), figure_height, inset=inset))
    if not segments:
        return
    fig.add_artist(
        LineCollection(
            segments,
            colors=[(0, 0, 0, YIELD_ALPHA)],
            linewidths=HATCH_LINEWIDTH,
            capstyle="butt",
            joinstyle="miter",
            # Window extents are already display pixels. At 72 dpi those pixels
            # are the SVG user units; dpi_scale_trans would scale them again.
            transform=IdentityTransform(),
            clip_on=False,
            zorder=5,
        )
    )


def plot_affinity_yield(frame: pd.DataFrame):
    colors = bar_colors(len(frame))
    x = np.arange(len(frame))
    width = 0.45

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=72)
    ax2 = ax1.twinx()

    bars_aff = ax1.bar(
        x - width / 2,
        frame["Affinity"],
        width,
        color=colors,
        edgecolor="none",
        zorder=3,
        clip_on=False,
    )
    bars_yield = ax2.bar(
        x + width / 2,
        frame["Yield"],
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=YIELD_ALPHA,
        zorder=3,
        clip_on=False,
    )

    for patch in bars_aff:
        height = patch.get_height()
        ax1.text(
            patch.get_x() + patch.get_width() / 2.0,
            height + 0.01,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=18,
        )

    for patch in bars_yield:
        height = patch.get_height()
        ax2.text(
            patch.get_x() + patch.get_width() / 2.0,
            height + 0.01,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=18,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(frame["PDB_ID"], rotation=30, ha="center", fontsize=20)
    ax1.set_ylabel("Affinity (nM)", fontsize=20)
    ax2.set_ylabel("Yield (mg)", fontsize=20)
    ax1.set_xlabel(None)
    ax1.tick_params(axis="y", labelsize=20)
    ax2.tick_params(axis="y", labelsize=20)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    legend_handles = [
        Patch(facecolor=LEGEND_COLOR, edgecolor="none", label="Affinity"),
        Patch(
            facecolor=LEGEND_COLOR,
            edgecolor="black",
            alpha=YIELD_ALPHA,
            label="Yield",
        ),
    ]
    legend = ax1.legend(handles=legend_handles, frameon=False, fontsize=20, loc="upper left")
    for handle in legend.legend_handles:
        handle.set_clip_on(False)

    fig.tight_layout()
    fig.canvas.draw()
    yield_legend = legend.legend_handles[1]
    add_yield_hatch(fig, [*bars_yield, yield_legend])
    return fig


def save_svg(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    svg = path.read_text(encoding="utf-8")
    if "<pattern" in svg or "clipPath" in svg or "clip-path" in svg:
        raise RuntimeError(f"{path} still contains a pattern or clip path")
    if "font-family: 'Arial'" not in svg:
        raise RuntimeError(f"{path} is missing an Arial text face")
    banned = (
        "Bitstream Vera Sans",
        "Computer Modern Sans Serif",
        "Geneva",
        "Lucid",
        "Avant Garde",
        "DejaVu Sans",
    )
    present = [name for name in banned if name in svg]
    if present:
        raise RuntimeError(f"{path} still names fonts Illustrator rejects: {present}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    figures = here.parents[2] / "figures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "Lag16_affinity_yield.txt",
        help="Tab- or whitespace-separated table with PDB_ID, Affinity, Yield.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=figures / "Lag16_affinity_yield_barplot.svg",
        help="SVG path. Text is stored as editable <text>, not outlined paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_affinity_yield(args.input)
    fig = plot_affinity_yield(frame)
    save_svg(fig, args.output)
    plt.close(fig)
    print(f"Read {len(frame)} rows from {args.input}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
