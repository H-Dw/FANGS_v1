#!/usr/bin/env python3
"""
Circle heatmaps of paired CDR-fragment insertions.

Revision notes:
- Legends were previously truncated or omitted. tight_layout is therefore
  omitted, and subplots_adjust reserves the right-hand margin.
- Plotting functions no longer depend on the global argument namespace.
- Color bins are ordered by ascending metric value, so smaller values
  occupy the leading groups: bottom 5%, 10%, 20%, 30%, 50%, and the remainder.
- Marker area uses an inverted mapping: smaller size-metric values are
  drawn as larger markers.
- The color legend is placed at the upper right and the size legend at the
  lower right.
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def parse_args():
    p = argparse.ArgumentParser(description="Circle heatmaps from CDR fragment PDB_IDs (color & size from two tables).")
    p.add_argument("--cdr_info", required=True, help="TSV containing PDBChain and CDR{n} sequences.")
    p.add_argument("--values_color", required=True, help="TSV with PDB_ID and color value.")
    p.add_argument("--values_size", required=True, help="TSV with PDB_ID and size value.")
    p.add_argument("--out_dir", required=True, help="Directory to write heatmaps.")
    p.add_argument("--cdr_indices", default="1,2,3", help="Comma-separated CDR indices (e.g. 1,2,3).")
    p.add_argument("--pdbid_col", default="PDB_ID", help="PDB ID column name used in both value tables.")
    p.add_argument("--color_col", default="selected_euclidean_average", help="Column name for color values in color TSV.")
    p.add_argument("--size_col", default="selected_euclidean_average", help="Column name for size values in size TSV.")
    p.add_argument("--per_chain", action="store_true", help="Also output per-chain heatmaps (default: aggregated only).")
    p.add_argument("--min_marker", type=float, default=10.0, help="Min scatter marker size (s).")
    p.add_argument("--max_marker", type=float, default=1000.0, help="Max scatter marker size (s).")
    p.add_argument("--size_legend_pct", default="10, 20, 30, 50, 100", help="Percentiles to show in the size legend, comma-separated (e.g. 10,50,90).")
    return p.parse_args()

def build_cdr_map(cdr_info_df):
    if 'PDBChain' not in cdr_info_df.columns:
        raise ValueError("cdr_info TSV must have column 'PDBChain'.")
    cdr_map = {}
    for _, row in cdr_info_df.iterrows():
        chain = str(row['PDBChain'])
        entry = {}
        for col in cdr_info_df.columns:
            if re.fullmatch(r'CDR\d+', col, flags=re.IGNORECASE):
                entry[col.upper()] = str(row[col]) if pd.notna(row[col]) else ""
        if entry:
            cdr_map[chain] = entry
    return cdr_map

def init_len_and_labels(seq):
    L = len(seq)
    list_len = max(1, 2 * L - 1) if L > 0 else 1
    char_pos = {}
    for i, ch in enumerate(seq):
        p = i * 2
        if p >= list_len:
            p = list_len - 1
        char_pos[p] = ch
    return list_len, char_pos

def pos_from_fragment(seq, frag, list_len):
    if not seq:
        return None
    s_upper = seq.upper()
    f_upper = frag.upper()
    start = s_upper.find(f_upper)
    if start < 0:
        return None
    end = start + len(frag) - 1
    middle = (start + end) / 2.0
    pos = int(round(middle * 2))
    pos = max(0, min(pos, list_len - 1))
    return pos

def prepare_pair_matrices(cdr_map, cdr_indices):
    dims = {}
    for chain, cdrs in cdr_map.items():
        dims[chain] = {}
        for ci in cdr_indices:
            cname = f'CDR{ci}'
            seq = cdrs.get(cname, "")
            L, labels = init_len_and_labels(seq)
            dims[chain][ci] = {'len': L, 'labels': labels, 'seq': seq}

    matrices_chain = {}
    # prepare only for ci != cj (skip identical combos)
    for chain in dims:
        for i in cdr_indices:
            for j in cdr_indices:
                if i == j:
                    continue
                li = dims[chain][i]['len']
                lj = dims[chain][j]['len']
                mat = [[[] for _ in range(lj)] for __ in range(li)]  # each cell holds list of (color,size) tuples
                matrices_chain[(chain, i, j)] = mat
    return dims, matrices_chain

def append_values_to_matrices(merged_df, cdr_map, dims, matrices_chain, cdr_indices, pdbid_col, color_col_merged, size_col_merged):
    pat = re.compile(r'CDR(\d+)-([A-Za-z]+)', flags=re.IGNORECASE)
    for _, row in merged_df.iterrows():
        pdbid = str(row[pdbid_col])
        try:
            color_val = float(row[color_col_merged])
            size_val = float(row[size_col_merged])
        except Exception:
            continue
        parts = pdbid.split("_")
        chain_hint = parts[1] if len(parts) > 1 else None

        # find all CDR-frag matches
        matches = []
        for m in pat.finditer(pdbid):
            ci = int(m.group(1))
            frag = m.group(2)
            if ci not in cdr_indices:
                continue
            matches.append((ci, frag))
        if not matches:
            continue

        resolved = []
        for ci, frag in matches:
            chosen_chain = None
            if chain_hint and chain_hint in cdr_map:
                chosen_chain = chain_hint
            else:
                for ch, cdrs in cdr_map.items():
                    seq = cdrs.get(f'CDR{ci}', "")
                    if seq and frag.upper() in seq.upper():
                        chosen_chain = ch
                        break
            if chosen_chain is None:
                continue
            seq = cdr_map[chosen_chain].get(f'CDR{ci}', "")
            list_len = dims[chosen_chain][ci]['len']
            pos = pos_from_fragment(seq, frag, list_len)
            if pos is None:
                continue
            resolved.append((ci, frag, chosen_chain, pos))
        if not resolved:
            continue

        # append color & size into matrices for each pair, skip identical-index combos
        for a in range(len(resolved)):
            for b in range(len(resolved)):
                ci, frag_i, chain_i, pos_i = resolved[a]
                cj, frag_j, chain_j, pos_j = resolved[b]
                if ci == cj:
                    continue
                key = (chain_i, ci, cj)
                if key not in matrices_chain:
                    continue
                mat = matrices_chain[key]
                if pos_i < 0 or pos_i >= len(mat):
                    continue
                if pos_j < 0 or pos_j >= len(mat[0]):
                    continue
                mat[pos_i][pos_j].append((color_val, size_val))

def aggregate_matrices(dims, matrices_chain, cdr_indices):
    per_chain_numeric = {}
    for (chain, ci, cj), mat in matrices_chain.items():
        li = len(mat)
        lj = len(mat[0]) if li > 0 else 0
        arr_color = np.full((li, lj), np.nan, dtype=float)
        arr_size  = np.full((li, lj), np.nan, dtype=float)
        for i in range(li):
            for j in range(lj):
                cell = mat[i][j]
                if cell:
                    colors = [t[0] for t in cell]
                    sizes  = [t[1] for t in cell]
                    arr_color[i,j] = float(np.mean(colors))
                    arr_size[i,j]  = float(np.mean(sizes))
        per_chain_numeric[(chain, ci, cj)] = (arr_color, arr_size)

    aggregated_color = {}
    aggregated_size = {}
    for ci in cdr_indices:
        for cj in cdr_indices:
            if ci == cj:
                aggregated_color[(ci,cj)] = None
                aggregated_size[(ci,cj)] = None
                continue
            arrs_c = []
            arrs_s = []
            max_li = 0
            max_lj = 0
            for (chain, a_ci, a_cj), (ac, asz) in per_chain_numeric.items():
                if a_ci == ci and a_cj == cj:
                    arrs_c.append(ac); arrs_s.append(asz)
                    max_li = max(max_li, ac.shape[0])
                    max_lj = max(max_lj, ac.shape[1])
            if not arrs_c:
                aggregated_color[(ci,cj)] = None
                aggregated_size[(ci,cj)] = None
                continue
            stack_c = [[[] for _ in range(max_lj)] for __ in range(max_li)]
            stack_s = [[[] for _ in range(max_lj)] for __ in range(max_li)]
            for ac, asz in zip(arrs_c, arrs_s):
                for i in range(ac.shape[0]):
                    for j in range(ac.shape[1]):
                        v = ac[i,j]
                        if not np.isnan(v):
                            stack_c[i][j].append(v)
                        sv = asz[i,j]
                        if not np.isnan(sv):
                            stack_s[i][j].append(sv)
            agg_c = np.full((max_li, max_lj), np.nan, dtype=float)
            agg_s = np.full((max_li, max_lj), np.nan, dtype=float)
            for i in range(max_li):
                for j in range(max_lj):
                    if stack_c[i][j]:
                        agg_c[i,j] = float(np.mean(stack_c[i][j]))
                    if stack_s[i][j]:
                        agg_s[i,j] = float(np.mean(stack_s[i][j]))
            aggregated_color[(ci,cj)] = agg_c
            aggregated_size[(ci,cj)]  = agg_s
    return (aggregated_color, aggregated_size), per_chain_numeric

def scale_marker_sizes_inverted(size_arr, min_s, max_s):
    """
    Map numeric size_arr -> marker sizes in [min_s, max_s].
    Inverted mapping: smaller input values -> larger markers.
    size_arr is 1D numpy array.
    """
    if size_arr.size == 0:
        return np.array([])
    smin = np.nanmin(size_arr)
    smax = np.nanmax(size_arr)
    if np.isclose(smin, smax):
        # constant input: return middle size
        return np.full(size_arr.shape, (min_s + max_s) / 2.0)
    # normalize 0..1
    norm = (size_arr - smin) / (smax - smin)
    # invert
    inv = 1.0 - norm
    scaled = inv * (max_s - min_s) + min_s
    return scaled

def plot_circle_heatmap(arr_color, arr_size, x_labels, y_labels, x_title, y_title, title, out_path,
                        min_marker=40.0, max_marker=600.0, color_label="color", size_label="size",
                        size_legend_pct_str="10,50,90"):

    if arr_color is None or arr_color.size == 0:
        print(f"Skipping {title}: no data.")
        return
    ny, nx = arr_color.shape

    # Collect data points
    xs, ys, colors_vals, sizes = [], [], [], []
    for i in range(ny):
        for j in range(nx):
            v = arr_color[i, j]
            sv = arr_size[i, j]
            if not np.isnan(v) and not np.isnan(sv):
                xs.append(j)
                ys.append(i)
                colors_vals.append(v)
                sizes.append(sv)
    if not colors_vals:
        print(f"Skipping {title}: empty after filtering.")
        return

    colors_vals = np.array(colors_vals)
    sizes = np.array(sizes)
    marker_sizes = scale_marker_sizes_inverted(sizes, min_marker, max_marker)

    # ----- Group by percentiles -----
    group_percentiles = [10, 20, 30, 50]
    group_colors = ["#D26546", "#EBBD76", "#986532", "#6EA486", "#3B5B8B"]

    percentile_values = {p: np.percentile(colors_vals, p) for p in group_percentiles}

    group_idx_for_value = []
    for v in colors_vals:
        assigned = None
        for idx, p in enumerate(group_percentiles):
            if v <= percentile_values[p]:
                assigned = idx
                break
        if assigned is None:
            assigned = len(group_colors) - 1  # Rest
        group_idx_for_value.append(assigned)
    group_idx_for_value = np.array(group_idx_for_value, dtype=int)
    point_colors = [group_colors[idx] for idx in group_idx_for_value]

    # ----- Create figure -----
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=False)

    sc = ax.scatter(xs, ys, c=point_colors, s=marker_sizes,
                    edgecolors='black', linewidths=0.35)

    # ----- Define Alignment Constants -----
    # Shared left edge of both legends in figure coordinates (range 0-1).
    LEGEND_X = 0.82 
    # Vertical positions of the color and size legends in figure coordinates.
    COLOR_LEGEND_Y = 0.85
    SIZE_LEGEND_Y = 0.50

    # ----- Build Color Legend -----
    label_map = [
        "Top 0–10%",
        "Top 10–20%",
        "Top 20–30%",
        "Top 30–50%",
        "Rest (>50%)"
    ]

    legend_patches = [
        Patch(facecolor=col, edgecolor='black') for col in group_colors
    ]

    # Anchor the legend at its upper-left corner and place that corner at LEGEND_X.
    color_legend = fig.legend(
        legend_patches, 
        label_map, 
        title=color_label.capitalize(),
        loc="upper left",                 # Anchor at the upper-left corner.
        bbox_to_anchor=(LEGEND_X, COLOR_LEGEND_Y), # (x, y)
        title_fontsize=14,
        fontsize=12,
        labelspacing=1.2,
        framealpha=0.0, borderaxespad=0., ncol=1
    )
    color_legend.get_title().set_fontweight("bold")

    # ----- Build Size Legend -----
    try:
        pct_list = [float(x) for x in size_legend_pct_str.split(",")]
    except:
        pct_list = [10.0, 50.0, 90.0]

    pct_vals = np.percentile(sizes, pct_list)
    pct_marker_sizes = scale_marker_sizes_inverted(pct_vals, min_marker, max_marker)

    size_handles = []
    size_labels_text = []
    
    for pct, ms in zip(pct_list, pct_marker_sizes):
        handle = Line2D([], [], marker='o', linestyle='None', color='gray',
                        markeredgecolor='black', markeredgewidth=0.4,
                        markersize=np.sqrt(ms))
        size_handles.append(handle)
        size_labels_text.append(f"Top {pct:.0f}%")
    
    size_legend = fig.legend(
        size_handles,
        size_labels_text,
        # label_map,
        title=size_label.capitalize(),
        loc="upper left",
        bbox_to_anchor=(LEGEND_X, SIZE_LEGEND_Y), 
        bbox_transform=fig.transFigure,
        framealpha=0.0,
        ncol=1,
        fontsize=12,
        title_fontsize=14,
        markerscale=1.0,
        labelspacing=1.8,
        handletextpad=0.6,
        handlelength=2.0,
        borderpad=0.4
    )
    size_legend.get_title().set_fontweight("bold")

    # ----- Axis formatting -----
    ax.set_xticks(range(nx))
    ax.set_yticks(range(ny))
    ax.set_xticklabels([x_labels.get(i, '') for i in range(nx)], fontsize=12)
    ax.set_yticklabels([y_labels.get(i, '') for i in range(ny)], fontsize=12)

    ax.set_xlabel(x_title, fontsize=14)
    ax.set_ylabel(y_title, fontsize=14)
    ax.set_title(title, fontsize=14)

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.invert_yaxis()

    # ----- Reserve space for legends -----
    fig.subplots_adjust(left=0.08, right=0.80, top=0.90, bottom=0.14)

    # ----- Save -----
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved circle-heatmap: {out_path}")

    
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cdr_info_df = pd.read_csv(args.cdr_info, sep="\t", dtype=str)
    df_color = pd.read_csv(args.values_color, sep="\t", dtype=str)
    df_size  = pd.read_csv(args.values_size, sep="\t", dtype=str)

    # check required columns exist
    if args.pdbid_col not in df_color.columns or args.pdbid_col not in df_size.columns:
        raise ValueError(f"PDB ID column '{args.pdbid_col}' must exist in both value tables.")
    if args.color_col not in df_color.columns:
        raise ValueError(f"Color TSV must contain column '{args.color_col}'.")
    if args.size_col not in df_size.columns:
        raise ValueError(f"Size TSV must contain column '{args.size_col}'.")

    # rename the color/size columns to avoid collision when merging
    color_col_merged = f"color_{args.color_col}"
    size_col_merged  = f"size_{args.size_col}"
    df_color_ren = df_color[[args.pdbid_col, args.color_col]].rename(columns={args.color_col: color_col_merged})
    df_size_ren  = df_size[[args.pdbid_col, args.size_col]].rename(columns={args.size_col: size_col_merged})

    # merge (inner join) on PDB_ID so each row has both a color and a size value
    merged = pd.merge(df_color_ren, df_size_ren, on=args.pdbid_col, how='inner')
    if merged.empty:
        raise ValueError("Merged table is empty after joining color and size tables on PDB_ID.")

    # convert to numeric and drop NaNs
    merged[color_col_merged] = pd.to_numeric(merged[color_col_merged], errors='coerce')
    merged[size_col_merged]  = pd.to_numeric(merged[size_col_merged], errors='coerce')
    merged = merged.dropna(subset=[color_col_merged, size_col_merged])
    if merged.empty:
        raise ValueError("No numeric rows remain after converting merged color/size to numeric.")

    cdr_map = build_cdr_map(cdr_info_df)
    cdr_indices = [int(x.strip()) for x in args.cdr_indices.split(",") if x.strip().isdigit()]

    dims, matrices_chain = prepare_pair_matrices(cdr_map, cdr_indices)

    append_values_to_matrices(merged, cdr_map, dims, matrices_chain, cdr_indices, args.pdbid_col, color_col_merged, size_col_merged)

    (aggregated_color, aggregated_size), per_chain_numeric = aggregate_matrices(dims, matrices_chain, cdr_indices)

    # Plot aggregated circle-heatmaps for each pair (skip identical ci==cj)
    for ci in cdr_indices:
        for cj in cdr_indices:
            if ci == cj:
                continue
            arr_c = aggregated_color.get((ci,cj))
            arr_s = aggregated_size.get((ci,cj))
            if arr_c is None or arr_s is None:
                continue
            # pick best chain for labels (largest area)
            best_chain = None
            best_area = 0
            for (chain, a_ci, a_cj), (ac, _) in per_chain_numeric.items():
                if a_ci==ci and a_cj==cj:
                    area = ac.shape[0]*ac.shape[1]
                    if area > best_area:
                        best_area = area; best_chain = chain
            if best_chain:
                x_labels = dims[best_chain][cj]['labels']
                y_labels = dims[best_chain][ci]['labels']
            else:
                x_labels = {}; y_labels = {}

            x_title = f"CDR{cj}"
            y_title = f"CDR{ci}"
            outname = f"{y_title}_vs_{x_title}_insertion_plot.png"
            outpath = os.path.join(args.out_dir, outname)
            
            plot_circle_heatmap(arr_c, arr_s, x_labels, y_labels, x_title, y_title,
                                title=None, out_path=outpath,
                                min_marker=args.min_marker, max_marker=args.max_marker,
                                color_label=os.path.basename(args.values_color).split('_')[0],
                                size_label=os.path.basename(args.values_size).split('_')[0], size_legend_pct_str=args.size_legend_pct)

    print("Finished all circle heatmaps.")

if __name__ == "__main__":
    main()