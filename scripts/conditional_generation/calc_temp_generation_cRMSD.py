"""Summarize CDR / whole-protein cRMSD and TM-score from TSV or PDB pairs.

Rows are filtered by pTM like ``extract_rf3_results.py`` (sort within each
``PDB_ID``, optional ``--min-ptm``, ``--top-k-per-pdb``, default top-1).
Writes ``ptm_filtered.tsv`` and ``statistics.txt`` in the RF3-style table format.

CDR cRMSD columns usually exist for every filtered row (from the input TSV).
``global_cRMSD`` / ``protein_TM_score`` are only filled when
``--template-pdb-dir`` and ``--generated-pdb-dir`` are provided and both PDBs
exist; counts often differ from 843 (see statistics file).

TM-score uses CA atoms, residue-wise correspondence, Kabsch superposition, and the
Zhang–Skolnick length-normalized formula (same setting as typical template–model
comparisons when numbering matches). No external binary is required.

When to use external tools instead: **US-align** (or TM-align) is preferable if you
need alignment *search* (different lengths, large insertions, domain rearrangements)
or a reference implementation’s exact TM definition. **Foldseek** is optimized for
massive structural search and 3Di-based similarity; it is not the usual choice for
reporting classical pairwise TM-score on two full chains with identical maps.
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Whole-protein cRMSD from PDB (CA atoms, Kabsch superposition) ------------

def parse_ca_from_pdb(pdb_path: str) -> dict[tuple[str, int], np.ndarray]:
    """Return {(chain_id, res_id): xyz} for CA atoms from a PDB file."""
    ca = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21]
            res_id = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            ca[(chain, res_id)] = np.array([x, y, z])
    return ca


def filter_ca_chain(
    ca: dict[tuple[str, int], np.ndarray], chain: str
) -> dict[tuple[str, int], np.ndarray]:
    return {k: v for k, v in ca.items() if k[0] == chain}


def _kabsch_superpose(
    ref_coords: np.ndarray, pred_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Center ref, rotate pred onto ref (same convention as cRMSD). Returns (ref_c, pred_aligned_c)."""
    assert ref_coords.shape == pred_coords.shape and ref_coords.shape[1] == 3
    c1 = ref_coords - ref_coords.mean(axis=0)
    c2 = pred_coords - pred_coords.mean(axis=0)
    h_mat = c1.T @ c2
    u_mat, _s, vt = np.linalg.svd(h_mat)
    d_det = np.linalg.det(vt.T @ u_mat.T)
    sign_matrix = np.diag([1, 1, np.sign(d_det)])
    rot = vt.T @ sign_matrix @ u_mat.T
    c2_rot = (rot @ c2.T).T
    return c1, c2_rot


def compute_crmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Coordinate RMSD after optimal superposition (same length, Nx3)."""
    c1, c2_rot = _kabsch_superpose(coords1, coords2)
    diff = c1 - c2_rot
    return float(np.sqrt((diff**2).sum(axis=1).mean()))


def d0_tm(length: int) -> float:
    """Zhang–Skolnick length-dependent scale (Å)."""
    le = max(int(length), 16)
    return float(1.24 * (le - 15.0) ** (1.0 / 3.0) - 1.8)


def compute_tm_score(
    coords1: np.ndarray,
    coords2: np.ndarray,
    length_norm: int | None = None,
) -> float:
    """TM-score after same Kabsch superposition as cRMSD (Zhang & Skolnick style).

    TM = (1/L) * sum_i 1/(1 + (d_i/d0)^2), with d_i CA distances after superposition.
    L defaults to the number of aligned residues (standard for full-chain comparison).
    """
    c1, c2_rot = _kabsch_superpose(coords1, coords2)
    d = np.linalg.norm(c1 - c2_rot, axis=1)
    l_target = int(length_norm) if length_norm is not None else len(d)
    d0 = d0_tm(l_target)
    d0 = max(d0, 0.5)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def paired_coords(
    ref_ca: dict[tuple[str, int], np.ndarray],
    pred_ca: dict[tuple[str, int], np.ndarray],
    keys: list[tuple[str, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if keys is None:
        keys = sorted(set(ref_ca) & set(pred_ca))
    shared = [k for k in keys if k in ref_ca and k in pred_ca]
    if not shared:
        raise ValueError("No shared CA residues found")
    ref = np.array([ref_ca[k] for k in shared])
    pred = np.array([pred_ca[k] for k in shared])
    return ref, pred


def chain_id_from_pdb_id(pdb_id: str) -> str:
    """e.g. '5u65A' -> 'A'; fallback 'A'."""
    if pdb_id and len(pdb_id) >= 1:
        return pdb_id[-1]
    return "A"


def compute_whole_protein_metrics(
    template_pdb: str,
    pred_pdb: str,
    chain: str | None = None,
) -> tuple[float | None, float | None]:
    """Whole-chain cRMSD and TM-score (same CA pairing and Kabsch superposition)."""
    if not os.path.isfile(template_pdb) or not os.path.isfile(pred_pdb):
        return None, None
    try:
        ref_ca = parse_ca_from_pdb(template_pdb)
        pred_ca = parse_ca_from_pdb(pred_pdb)
        if chain:
            ref_ca = filter_ca_chain(ref_ca, chain)
            pred_ca = filter_ca_chain(pred_ca, chain)
        ref_coords, pred_coords = paired_coords(ref_ca, pred_ca)
        crmsd = compute_crmsd(ref_coords, pred_coords)
        tm = compute_tm_score(ref_coords, pred_coords, length_norm=len(ref_coords))
        return crmsd, tm
    except (ValueError, OSError, KeyError):
        return None, None


def normalize_protein_crmsd_column(data: pd.DataFrame) -> pd.DataFrame:
    """Use protein_cRMSD, or copy global_cRMSD if present."""
    out = data.copy()
    if "protein_cRMSD" not in out.columns and "global_cRMSD" in out.columns:
        out["protein_cRMSD"] = out["global_cRMSD"]
    return out


def normalize_protein_tm_column(data: pd.DataFrame) -> pd.DataFrame:
    """Use protein_TM_score if missing; accept TM_score / tm_score from upstream TSV."""
    out = data.copy()
    if "protein_TM_score" not in out.columns:
        for alt in ("TM_score", "tm_score", "TM-score"):
            if alt in out.columns:
                out["protein_TM_score"] = out[alt]
                break
    return out


def add_protein_metrics_from_structure_dirs(
    data: pd.DataFrame,
    template_pdb_dir: str,
    generated_pdb_dir: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add protein_cRMSD and protein_TM_score (same superposition) per row.

    Returns counts explaining how many rows lack whole-protein metrics (template /
    prediction missing on disk, or alignment failure). CDR columns from the TSV are
    unchanged and typically still have one value per filtered row.
    """
    out = data.copy()
    rows_crmsd = []
    rows_tm = []
    missing_tpl = 0
    missing_pred = 0
    failed = 0
    empty_filename = 0
    for idx, row in out.iterrows():
        pdb_id = row.get("PDB_ID", "")
        fname = row.get("Filename", "")
        if pd.isna(fname) or not str(fname).strip():
            rows_crmsd.append(np.nan)
            rows_tm.append(np.nan)
            empty_filename += 1
            continue
        chain = chain_id_from_pdb_id(str(pdb_id))
        tpl = os.path.join(template_pdb_dir, f"{pdb_id}.pdb")
        pred = os.path.join(generated_pdb_dir, str(fname).strip())
        if not os.path.isfile(tpl):
            missing_tpl += 1
            rows_crmsd.append(np.nan)
            rows_tm.append(np.nan)
            continue
        if not os.path.isfile(pred):
            missing_pred += 1
            rows_crmsd.append(np.nan)
            rows_tm.append(np.nan)
            continue
        crmsd, tm = compute_whole_protein_metrics(tpl, pred, chain=chain)
        if crmsd is None:
            failed += 1
        rows_crmsd.append(crmsd)
        rows_tm.append(tm)
    out["protein_cRMSD"] = rows_crmsd
    out["protein_TM_score"] = rows_tm
    success = int(pd.Series(rows_crmsd).notna().sum())
    counts = {
        "missing_template": missing_tpl,
        "missing_prediction": missing_pred,
        "alignment_failed": failed,
        "empty_filename": empty_filename,
        "success_whole_protein": success,
    }
    print(
        f"whole-protein metrics: success={success}, missing_template={missing_tpl}, "
        f"missing_prediction={missing_pred}, alignment_failed={failed}, "
        f"empty_filename={empty_filename}",
        file=sys.stderr,
    )
    return out, counts


def filter_by_ptm(
    data: pd.DataFrame,
    min_ptm: float | None = None,
    top_k_per_pdb: int | None = 1,
    id_col: str = "PDB_ID",
) -> tuple[pd.DataFrame, int]:
    """Sort by pTM within each PDB_ID, optional min_ptm, keep top-K per id (extract_rf3 style).

    Sets ``ptm_rank_in_target`` to 1..K within each ``id_col`` among rows kept.
    """
    df = data.copy()
    if "pTM" not in df.columns and "ptm" in df.columns:
        df = df.rename(columns={"ptm": "pTM"})
    if id_col not in df.columns:
        raise ValueError(f"Input TSV must contain column {id_col!r}")
    if "pTM" not in df.columns:
        raise ValueError("Input TSV must contain column 'pTM' or 'ptm'")

    n_before = len(df)
    df = df.sort_values(by=[id_col, "pTM"], ascending=[True, False], na_position="last")

    if min_ptm is not None:
        df = df[df["pTM"].notna() & (df["pTM"] >= float(min_ptm))]

    df["ptm_rank_in_target"] = df.groupby(id_col, sort=False).cumcount() + 1

    if top_k_per_pdb is not None and top_k_per_pdb > 0:
        df = df[df["ptm_rank_in_target"] <= top_k_per_pdb]

    dropped = n_before - len(df)
    return df.reset_index(drop=True), dropped


def parse_exclude_protein_ids(s: str | None) -> frozenset[str]:
    """Parse comma/whitespace-separated IDs (e.g. ``7pklL,8jbhE``). Case-sensitive."""
    if not s or not str(s).strip():
        return frozenset()
    parts = re.split(r"[\s,]+", str(s).strip())
    return frozenset(p for p in parts if p)


def apply_exclude_protein_ids(
    data: pd.DataFrame,
    exclude: frozenset[str],
    id_col: str = "PDB_ID",
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose ``id_col`` equals an excluded ID or shares the same leading token (before ``_``)."""
    if not exclude:
        return data, 0
    if id_col not in data.columns:
        raise ValueError(f"Input TSV must contain column {id_col!r}")
    s = data[id_col].astype(str)
    base = s.str.split("_").str[0]
    mask = ~(s.isin(exclude) | base.isin(exclude))
    n_removed = int((~mask).sum())
    return data.loc[mask].reset_index(drop=True), n_removed


def align_columns_with_rf3_extract(data: pd.DataFrame) -> pd.DataFrame:
    """Add ``global_cRMSD`` alias for whole-protein cRMSD (same as extract_rf3_results)."""
    out = data.copy()
    if "global_cRMSD" not in out.columns and "protein_cRMSD" in out.columns:
        out["global_cRMSD"] = out["protein_cRMSD"]
    return out


def write_statistics_rf3_style(
    df: pd.DataFrame,
    stats_path: str,
    *,
    title: str,
    min_ptm: float | None,
    top_k_per_pdb: int | None,
    rows_dropped_by_ptm: int,
    protein_counts: dict[str, int] | None,
    id_col: str = "PDB_ID",
    excluded_protein_ids: frozenset[str] | None = None,
    rows_removed_by_exclusion: int = 0,
) -> None:
    """Same layout as ``extract_rf3_results.py`` (metric table + per-id means)."""
    cdr_cols = sorted([c for c in df.columns if c.startswith("CDR") and "cRMSD" in c])
    metric_cols = ["global_cRMSD", "protein_TM_score"] + cdr_cols + ["plddt", "pTM"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    with open(stats_path, "w") as f:
        f.write(f"{title}\n")
        f.write(
            "(Metrics on CDR / pTM are for all rows after pTM filtering; "
            "global_cRMSD / protein_TM_score only where PDBs were found and aligned.)\n"
        )
        f.write("=" * 50 + "\n\n")
        if excluded_protein_ids:
            f.write(
                f"Excluded protein IDs: {', '.join(sorted(excluded_protein_ids))}\n"
                f"Rows removed by exclusion (before pTM selection): {rows_removed_by_exclusion}\n\n"
            )
        f.write(f"Selection: min_ptm={min_ptm}, top_k_per_pdb={top_k_per_pdb}\n")
        f.write(f"Rows dropped by pTM filter: {rows_dropped_by_ptm}\n\n")
        f.write(f"Total samples (after filter): {len(df)}\n")
        f.write(f"Total {id_col}s: {df[id_col].nunique()}\n\n")

        if protein_counts:
            f.write("--- Whole-protein metrics coverage (global_cRMSD / protein_TM_score) ---\n")
            f.write(
                f"  Rows with successful alignment: {protein_counts.get('success_whole_protein', 0)}\n"
                f"  Missing template PDB: {protein_counts.get('missing_template', 0)}\n"
                f"  Missing prediction PDB: {protein_counts.get('missing_prediction', 0)}\n"
                f"  Alignment / parse failed: {protein_counts.get('alignment_failed', 0)}\n"
                f"  Empty Filename column: {protein_counts.get('empty_filename', 0)}\n"
            )
            f.write(
                "\nWhy CDR count can exceed whole-protein count: CDR cRMSD comes from the "
                "input TSV for every filtered row; whole-protein cRMSD is computed only "
                "when both template `{{PDB_ID}}.pdb` and generated `Filename` exist under "
                "the given directories and CA pairing succeeds.\n\n"
            )
        else:
            f.write(
                "\n(Whole-protein metrics were not computed: pass --template-pdb-dir and "
                "--generated-pdb-dir to fill global_cRMSD / protein_TM_score.)\n\n"
            )

        f.write(f"{'Metric':<20s} {'Mean':>10s} {'Std':>10s} {'Median':>10s} {'Min':>10s} {'Max':>10s}\n")
        f.write("-" * 70 + "\n")

        for col in metric_cols:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            f.write(
                f"{col:<20s} {vals.mean():>10.4f} {vals.std():>10.4f} "
                f"{vals.median():>10.4f} {vals.min():>10.4f} {vals.max():>10.4f}\n"
            )

        if id_col in df.columns:
            f.write(f"\n\n--- Per-{id_col} mean statistics ---\n\n")
            gcols = [c for c in metric_cols if c in df.columns]
            target_means = df.groupby(id_col)[gcols].mean()
            f.write(f"{'Metric':<20s} {'Mean':>10s} {'Std':>10s} {'Median':>10s} {'Min':>10s} {'Max':>10s}\n")
            f.write("-" * 70 + "\n")
            for col in gcols:
                vals = target_means[col].dropna()
                if len(vals) == 0:
                    continue
                f.write(
                    f"{col:<20s} {vals.mean():>10.4f} {vals.std():>10.4f} "
                    f"{vals.median():>10.4f} {vals.min():>10.4f} {vals.max():>10.4f}\n"
                )

    print(f"Statistics saved to: {stats_path}")


def plot_boxplot_with_errorbar(data, output_dir):
    cdr_columns = ["CDR1_cRMSD", "CDR2_cRMSD", "CDR3_cRMSD"]
    has_protein = "protein_cRMSD" in data.columns and data["protein_cRMSD"].notna().any()

    melt_cols = list(cdr_columns)
    rename_map = {
        "CDR1_cRMSD": "CDR1",
        "CDR2_cRMSD": "CDR2",
        "CDR3_cRMSD": "CDR3",
    }
    if has_protein:
        melt_cols.append("protein_cRMSD")
        rename_map["protein_cRMSD"] = "Whole protein"

    long_data = data[melt_cols].melt(var_name="Region", value_name="cRMSD Value")
    long_data["Region"] = long_data["Region"].replace(rename_map)

    order = ["CDR1", "CDR2", "CDR3"] + (["Whole protein"] if has_protein else [])
    palette = (
        ["#7AB656", "#7E99F4", "#CC7C71", "#B565A7"]
        if has_protein
        else ["#7AB656", "#7E99F4", "#CC7C71"]
    )

    plt.figure(figsize=(12 if has_protein else 10, 6))
    sns.boxplot(
        data=long_data,
        x="Region",
        y="cRMSD Value",
        order=order,
        hue="Region",
        palette=palette,
        legend=False,
    )
    plt.ylabel("cRMSD", fontsize=14, fontweight="bold")

    output_plot_path = os.path.join(output_dir, "cdr_rmsd_boxplot_with_errorbar.png")
    plt.savefig(output_plot_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Boxplot with error bars saved to: {output_plot_path}")


def plot_protein_tm_distribution(data, output_dir):
    if "protein_TM_score" not in data.columns or not data["protein_TM_score"].notna().any():
        return
    tmv = data["protein_TM_score"].dropna()
    plt.figure(figsize=(8, 5))
    sns.histplot(tmv, bins=30, kde=True, color="#2C5F8D")
    plt.xlabel("TM-score (whole protein)", fontsize=12, fontweight="bold")
    plt.ylabel("Count", fontsize=12, fontweight="bold")
    plt.xlim(0.0, 1.0)
    out_path = os.path.join(output_dir, "protein_tm_score_distribution.png")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"TM-score histogram saved to: {out_path}")


def save_processed_data(data: pd.DataFrame, output_dir: str) -> None:
    """Write pTM-filtered table (tab-separated), same role as extract_rf3_results TSV."""
    path = os.path.join(output_dir, "ptm_filtered.tsv")
    data.to_csv(path, index=False, sep="\t")
    print(f"pTM-filtered TSV saved to: {path}")
    legacy = os.path.join(output_dir, "processed_data.csv")
    data.to_csv(legacy, index=False, sep="\t")
    print(f"(Legacy copy, same content: {legacy})")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize CDR and whole-protein cRMSD / TM-score from a TSV; optionally "
            "compute protein_cRMSD and protein_TM_score from template and generated PDBs "
            "(same CA alignment; TM-score uses Zhang–Skolnick formula, no external tools). "
            "pTM filtering matches extract_rf3_results.py (sort desc, min_ptm, top_k per id)."
        )
    )
    parser.add_argument("input_file", type=str, help="Path to the input TSV file")
    parser.add_argument("output_dir", type=str, help="Directory to save the output files")
    parser.add_argument(
        "--template-pdb-dir",
        type=str,
        default=None,
        help="Directory of template PDBs named {PDB_ID}.pdb (e.g. 5u65A.pdb). "
        "With --generated-pdb-dir, computes protein_cRMSD and protein_TM_score per row.",
    )
    parser.add_argument(
        "--generated-pdb-dir",
        type=str,
        default=None,
        help="Directory containing files listed in the Filename column (generated structures).",
    )
    parser.add_argument(
        "--min-ptm",
        type=float,
        default=None,
        metavar="P",
        help="Keep only rows with pTM >= P (after sorting by pTM within each PDB_ID).",
    )
    parser.add_argument(
        "--top-k-per-pdb",
        type=int,
        default=1,
        metavar="K",
        help="Keep at most K rows per PDB_ID (highest pTM first). Default 1 matches "
        "extract_rf3 top-1 per target. Use 0 to keep all rows after --min-ptm (no cap).",
    )
    parser.add_argument(
        "--exclude-protein-ids",
        type=str,
        default="",
        metavar="IDS",
        help="Comma or space separated protein/PDB IDs to remove before pTM selection "
        "and statistics (e.g. 7pklL,8jbhE). Matches full PDB_ID or the prefix before '_'.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw = pd.read_csv(args.input_file, sep="\t")
    exclude_ids = parse_exclude_protein_ids(args.exclude_protein_ids)
    raw, n_excluded = apply_exclude_protein_ids(raw, exclude_ids, id_col="PDB_ID")
    if exclude_ids:
        print(
            f"Excluded {n_excluded} row(s); IDs: {', '.join(sorted(exclude_ids))}",
            file=sys.stderr,
        )
    if len(raw) == 0:
        print("ERROR: no rows left after --exclude-protein-ids", file=sys.stderr)
        sys.exit(1)

    processed_data, dropped_ptm = filter_by_ptm(
        raw,
        min_ptm=args.min_ptm,
        top_k_per_pdb=args.top_k_per_pdb,
        id_col="PDB_ID",
    )
    processed_data = normalize_protein_crmsd_column(processed_data)
    processed_data = normalize_protein_tm_column(processed_data)

    tpl_dir = args.template_pdb_dir
    gen_dir = args.generated_pdb_dir
    protein_counts: dict[str, int] | None = None
    if (tpl_dir is None) ^ (gen_dir is None):
        print(
            "ERROR: provide both --template-pdb-dir and --generated-pdb-dir, or neither.",
            file=sys.stderr,
        )
        sys.exit(1)
    if tpl_dir and gen_dir:
        processed_data, protein_counts = add_protein_metrics_from_structure_dirs(
            processed_data, tpl_dir, gen_dir
        )
    else:
        missing = []
        if "protein_cRMSD" not in processed_data.columns:
            missing.append("protein_cRMSD")
        if "protein_TM_score" not in processed_data.columns:
            missing.append("protein_TM_score")
        if missing:
            print(
                f"Note: no {' / '.join(missing)}; use --template-pdb-dir + "
                "--generated-pdb-dir or add columns to the TSV. "
                "CDR cRMSD and pTM are still summarized.",
                file=sys.stderr,
            )

    processed_data = align_columns_with_rf3_extract(processed_data)

    stats_path = os.path.join(args.output_dir, "statistics.txt")
    write_statistics_rf3_style(
        processed_data,
        stats_path,
        title="Template Generation Results - Statistical Summary",
        min_ptm=args.min_ptm,
        top_k_per_pdb=args.top_k_per_pdb,
        rows_dropped_by_ptm=dropped_ptm,
        protein_counts=protein_counts,
        id_col="PDB_ID",
        excluded_protein_ids=exclude_ids if exclude_ids else None,
        rows_removed_by_exclusion=n_excluded,
    )

    save_processed_data(processed_data, args.output_dir)
    plot_boxplot_with_errorbar(processed_data, args.output_dir)
    plot_protein_tm_distribution(processed_data, args.output_dir)


if __name__ == "__main__":
    main()
