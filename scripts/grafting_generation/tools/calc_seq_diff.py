import argparse
import sys
import shutil
import subprocess
import io
import os
import numpy as np
import pandas as pd
import matplotlib

# SVG text must stay as <text> nodes. The default "path" fonttype outlines
# every glyph, so a number or label cannot be edited as one string.
matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "axes.formatter.use_mathtext": False,
    }
)
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from mpl_toolkits.axes_grid1 import make_axes_locatable

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze pairwise sequence differences, with optional MAFFT alignment, custom order, label simplification, and colormap selection."
    )
    
    # Basic input/output
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file path")
    parser.add_argument("-o", "--output", required=True, help="Output file prefix")
    
    # MAFFT options
    parser.add_argument("-m", "--mafft", nargs='?', const="mafft", default=None,
                        help="Run MAFFT alignment. Optionally specify the MAFFT executable path. If omitted, uses 'mafft' from PATH.")
    parser.add_argument("--save-aligned", action="store_true", default=True,
                        help="Save the aligned FASTA file (default: True)")
    
    # Plot content
    parser.add_argument("--order", help="Path to a list file of IDs (one per line) that sets heatmap row/column order.")
    parser.add_argument("--simplify", action="store_true", 
                        help="Simplify labels: use only the first '_' -delimited token as the display name (e.g. GeneA_Species_01 -> GeneA).")
    
    # Plot style: cell size (inches) vs font size (points) controls the visual ratio.
    # Smaller --cell-size makes squares smaller relative to numbers/labels.
    # Larger --fontsize / --annot-size / --text-scale makes text larger relative to squares.
    parser.add_argument("--cell-size", type=float, default=0.45,
                        help="Size of each heatmap cell in inches (default: 0.45). Decrease this if squares look too large relative to numbers and labels.")
    parser.add_argument("--fontsize", type=int, default=20,
                        help="Heatmap axis tick label font size in points (default: 20).")
    parser.add_argument("--annot-size", type=int, default=20,
                        help="Font size of numbers inside heatmap cells in points (default: 20).")
    parser.add_argument("--text-scale", type=float, default=1.0,
                        help="Scale factor applied to both tick labels and cell numbers (default: 1.0). Increase this to make text larger without changing cell size.")
    parser.add_argument("--color", choices=['blue-green', 'red', 'viridis', 'magma', 'blue'], default='blue-green',
                        help="Heatmap colormap: blue-green (default), red, viridis, magma, or blue.")
    parser.add_argument("--format", choices=["png", "svg", "both"], default="both",
                        help="Heatmap image format (default: both). SVG stores tick labels, cell numbers, and the colorbar label as editable <text> elements.")

    return parser.parse_args()

def run_mafft(input_file, mafft_path="mafft"):
    """Run MAFFT and return the alignment as a string."""
    if not shutil.which(mafft_path):
        print(f"[ERROR] MAFFT executable not found: '{mafft_path}'")
        sys.exit(1)

    print(f"[INFO] Running MAFFT ({mafft_path})...")
    cmd = [mafft_path, "--auto", "--quiet", input_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] MAFFT failed:\n{e.stderr}")
        sys.exit(1)

def calculate_diff(seq1, seq2):
    """Count pairwise differences."""
    len1, len2 = len(seq1), len(seq2)
    min_len = min(len1, len2)
    diff_count = abs(len1 - len2)
    
    for k in range(min_len):
        if seq1[k] != seq2[k]:
            diff_count += 1
    return diff_count

def get_color_map(color_choice):
    """Map a user-friendly color name to a Matplotlib colormap."""
    maps = {
        'red': 'Reds',
        'blue-green': 'GnBu',  # Green to Blue
        'blue': 'Blues',
        'viridis': 'viridis',
        'magma': 'magma'
    }
    return maps.get(color_choice, 'GnBu')

def save_heatmap(fig, prefix, image_format):
    """Write the heatmap. SVG text is one <text> node per label or number."""
    formats = ["png", "svg"] if image_format == "both" else [image_format]
    for fmt in formats:
        path = f"{prefix}_heatmap.{fmt}"
        if fmt == "png":
            fig.savefig(path, dpi=600, bbox_inches="tight", format="png")
            print(f"[INFO] Heatmap saved to: {path} (DPI 600)")
        else:
            # Omit dpi so labels are not baked into a raster image.
            fig.savefig(path, bbox_inches="tight", format="svg")
            print(f"[INFO] Heatmap saved to: {path} (editable SVG text)")

def main():
    args = parse_args()
    
    # --- 1. Load / align sequences ---
    records = []
    
    if args.mafft:
        aligned_str = run_mafft(args.input, args.mafft)
        
        # Save alignment
        if args.save_aligned:
            aln_filename = f"{args.output}_aligned.fasta"
            with open(aln_filename, "w") as f:
                f.write(aligned_str)
            print(f"[INFO] Aligned sequences saved to: {aln_filename}")
            
        fasta_io = io.StringIO(aligned_str)
        records = list(SeqIO.parse(fasta_io, "fasta"))
    else:
        print(f"[INFO] Reading input file: {args.input}")
        records = list(SeqIO.parse(args.input, "fasta"))

    if len(records) < 2:
        print("[ERROR] Not enough sequences to analyze.")
        sys.exit(1)

    # Extract original IDs and sequences
    # Store in a dict for later reordering
    id_seq_map = {rec.id: str(rec.seq).upper() for rec in records}
    current_ids = list(id_seq_map.keys())
    
    # --- 2. Apply custom order ---
    if args.order:
        print(f"[INFO] Reading order file: {args.order}")
        try:
            with open(args.order, 'r') as f:
                # Read non-empty lines and strip whitespace
                ordered_ids = [line.strip() for line in f if line.strip()]
            
            # Keep only IDs that exist in the FASTA
            final_ids = [uid for uid in ordered_ids if uid in id_seq_map]
            
            # Check for missing IDs
            missing_in_fasta = set(ordered_ids) - set(id_seq_map.keys())
            missing_in_order = set(id_seq_map.keys()) - set(ordered_ids)
            
            if missing_in_fasta:
                print(f"[WARNING] Order file contains IDs not in the FASTA (ignored): {len(missing_in_fasta)}")
            if missing_in_order:
                print(f"[WARNING] FASTA contains IDs not listed in the order file (appended at the end): {len(missing_in_order)}")
                final_ids.extend(list(missing_in_order))
            
            current_ids = final_ids
            
        except Exception as e:
            print(f"[ERROR] Failed to read order file: {e}")
            sys.exit(1)

    n = len(current_ids)
    print(f"[INFO] Analyzing {n} sequences.")

    # --- 3. Compute difference matrix ---
    # Build the matrix in current_ids order
    diff_matrix = np.zeros((n, n), dtype=int)
    seqs_ordered = [id_seq_map[uid] for uid in current_ids]

    for i in range(n):
        for j in range(i + 1, n):
            d = calculate_diff(seqs_ordered[i], seqs_ordered[j])
            diff_matrix[i, j] = d
            diff_matrix[j, i] = d
            
    # Create DataFrame
    df = pd.DataFrame(diff_matrix, index=current_ids, columns=current_ids)
    
    # --- 4. Simplify labels ---
    if args.simplify:
        print("[INFO] Simplifying labels (using the token before the first '_')...")
        # Build rename map: {old_name: new_name}
        rename_map = {}
        for uid in current_ids:
            new_name = uid.split('_')[0]
            rename_map[uid] = new_name
        
        # Check for duplicate names after simplification
        if len(set(rename_map.values())) < len(rename_map):
            print("[WARNING] Label simplification produced duplicate names; the heatmap will show identical labels.")
            
        df.rename(index=rename_map, columns=rename_map, inplace=True)

    # Save matrix
    out_csv = f"{args.output}_matrix.csv"
    df.to_csv(out_csv)
    print(f"[INFO] Difference matrix saved to: {out_csv}")

    # --- 5. Draw heatmap ---
    print(f"[INFO] Drawing heatmap (colormap: {args.color})...")

    tick_fontsize = args.fontsize * args.text_scale
    annot_fontsize = args.annot_size * args.text_scale
    cell_size = args.cell_size

    # Extra margins so tick labels are not clipped (char width ~0.55 of fontsize)
    max_tick_len = max((len(str(x)) for x in list(df.index) + list(df.columns)), default=1)
    y_label_inch = max(0.9, max_tick_len * tick_fontsize * 0.55 / 72.0)
    if args.simplify:
        x_label_inch = max(0.7, tick_fontsize * 1.4 / 72.0)
    else:
        x_label_inch = max(1.1, max_tick_len * tick_fontsize * 0.40 / 72.0)
    cbar_inch = 0.85

    fig_w = n * cell_size + y_label_inch + cbar_inch
    fig_h = n * cell_size + x_label_inch
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Show cell annotations only for smaller matrices
    annot = True if n <= 30 else False
    
    cmap_name = get_color_map(args.color)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    
    # Plot
    sns.heatmap(
        df, 
        ax=ax,
        annot=annot, 
        fmt="d", 
        cmap=cmap_name, 
        annot_kws={
            "size": annot_fontsize,
            "fontweight": "bold"
        },
        square=True,
        linewidths=1.5,
        cbar=True,
        cbar_ax=cax,
        cbar_kws={'label': 'Number of AA Differences'})

    # Title
    # title_str = "Pairwise AA Differences"
    # if args.mafft:
    #     title_str += " (Aligned)"
    # plt.title(title_str, fontsize=tick_fontsize + 4)
    
    # Axis ticks
    ax.set_xticklabels(ax.get_xticklabels(),
                    # rotation=45 if not args.simplify else 0,
                    rotation=30,
                    ha='right' if not args.simplify else 'center',
                    fontsize=tick_fontsize)

    ax.set_yticklabels(ax.get_yticklabels(),
                    rotation=0,
                    fontsize=tick_fontsize)

    # Colorbar fonts
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label('Number of AA Differences', fontsize=tick_fontsize)
    cbar.ax.yaxis.labelpad = 8

    # Plain strings, not mathtext. Mathtext is written as one <tspan> per glyph.
    for text in list(ax.texts) + list(cbar.ax.texts):
        text.set_usetex(False)
    for axis in (ax.xaxis, ax.yaxis, cbar.ax.xaxis, cbar.ax.yaxis):
        for label in axis.get_ticklabels():
            label.set_usetex(False)

    plt.tight_layout()
    save_heatmap(fig, args.output, args.format)
    plt.close(fig)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
