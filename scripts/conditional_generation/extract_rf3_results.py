#!/usr/bin/env python3
"""Extract and analyze RF3 prediction results.

For each target, collects all seeds, sorts by pTM (descending), optionally keeps
only seeds above --min-ptm and/or the top --top-k-per-target, then computes
template-aligned cRMSD (global + per-CDR) and whole-protein TM-score (same
Kabsch + Zhang–Skolnick definition as calc_temp_generation_cRMSD.py).

Writes a TSV and a statistics summary for the filtered set only.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_exclude_protein_ids(s: str | None) -> frozenset[str]:
    """Comma/whitespace-separated IDs; matches prediction folder name or its leading token (before ``_``)."""
    if not s or not str(s).strip():
        return frozenset()
    parts = re.split(r"[\s,]+", str(s).strip())
    return frozenset(p for p in parts if p)


# ---------------------------------------------------------------------------
# Structural I/O helpers
# ---------------------------------------------------------------------------

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


def parse_ca_from_cif(cif_path: str) -> dict[tuple[str, int], np.ndarray]:
    """Return {(auth_chain, auth_seq_id): xyz} for CA atoms from an mmCIF file."""
    ca = {}
    in_atom_site = False
    columns: list[str] = []

    with open(cif_path) as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("_atom_site."):
                in_atom_site = True
                col_name = stripped.split(".")[1].strip()
                columns.append(col_name)
                continue

            if in_atom_site and not stripped.startswith("ATOM") and not stripped.startswith("HETATM"):
                if stripped.startswith("_") or stripped.startswith("#") or stripped.startswith("loop_"):
                    in_atom_site = False
                    columns = []
                continue

            if not in_atom_site:
                continue

            fields = stripped.split()
            if len(fields) < len(columns):
                continue

            col_map = {c: fields[i] for i, c in enumerate(columns)}

            if col_map.get("group_PDB") != "ATOM":
                continue
            if col_map.get("auth_atom_id", col_map.get("label_atom_id", "")) != "CA":
                continue

            chain = col_map.get("auth_asym_id", col_map.get("label_asym_id", "A"))
            try:
                res_id = int(col_map.get("auth_seq_id", col_map.get("label_seq_id", "0")))
            except ValueError:
                continue

            x = float(col_map["Cartn_x"])
            y = float(col_map["Cartn_y"])
            z = float(col_map["Cartn_z"])
            ca[(chain, res_id)] = np.array([x, y, z])

    return ca


# ---------------------------------------------------------------------------
# RMSD computation (no superposition – coordinate RMSD)
# ---------------------------------------------------------------------------

def _kabsch_superpose(
    ref_coords: np.ndarray, pred_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Center ref, rotate pred onto ref. Returns (ref_c, pred_aligned_c)."""
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
    """Coordinate RMSD after Kabsch superposition (same as calc_temp_generation_cRMSD)."""
    c1, c2_rot = _kabsch_superpose(coords1, coords2)
    diff = c1 - c2_rot
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def d0_tm(length: int) -> float:
    """Zhang–Skolnick length-dependent scale (Å)."""
    le = max(int(length), 16)
    return float(1.24 * (le - 15.0) ** (1.0 / 3.0) - 1.8)


def compute_tm_score(
    coords1: np.ndarray,
    coords2: np.ndarray,
    length_norm: int | None = None,
) -> float:
    """TM-score after same Kabsch superposition as cRMSD (Zhang & Skolnick)."""
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
    """Extract paired coordinate arrays for shared residue keys."""
    if keys is None:
        keys = sorted(set(ref_ca) & set(pred_ca))
    shared = [k for k in keys if k in ref_ca and k in pred_ca]
    if not shared:
        raise ValueError("No shared CA residues found")
    ref = np.array([ref_ca[k] for k in shared])
    pred = np.array([pred_ca[k] for k in shared])
    return ref, pred


# ---------------------------------------------------------------------------
# template_selection parser
# ---------------------------------------------------------------------------

def parse_template_selection(selections: list[str]) -> list[tuple[str, int, int]]:
    """Parse RF3 AtomSelection strings like 'A/*/26-33' -> (chain, start, end)."""
    regions = []
    for sel in selections:
        parts = sel.split("/")
        chain = parts[0]
        res_range = parts[2]
        m = re.match(r"(\d+)-(\d+)", res_range)
        if m:
            regions.append((chain, int(m.group(1)), int(m.group(2))))
    return regions


def region_keys(chain: str, start: int, end: int) -> list[tuple[str, int]]:
    """Generate (chain, res_id) keys for a residue range (inclusive)."""
    return [(chain, r) for r in range(start, end + 1)]


def _collect_target_context(
    configs_dir: str,
    target_id: str,
) -> tuple[dict | None, dict[tuple[str, int], np.ndarray], list[tuple[str, int, int]], str | None]:
    """Load template CA and CDR regions for a target. Returns (None, ...) on failure."""
    config_path = os.path.join(configs_dir, f"{target_id}_config.json")
    if not os.path.isfile(config_path):
        return None, {}, [], f"{target_id}: config not found at {config_path}"
    with open(config_path) as f:
        config = json.load(f)
    cfg = config[0] if isinstance(config, list) else config
    template_path = cfg["components"][0]["path"]
    template_selections = cfg.get("template_selection", [])
    if not os.path.isfile(template_path):
        return None, {}, [], f"{target_id}: template PDB not found at {template_path}"
    ref_ca = parse_ca_from_pdb(template_path)
    cdr_regions = parse_template_selection(template_selections)
    meta = {"template_path": template_path, "cfg": cfg}
    return meta, ref_ca, cdr_regions, None


def _filter_seeds_by_ptm(
    candidates: list[dict],
    min_ptm: float | None,
    top_k_per_target: int | None,
) -> tuple[list[dict], int]:
    """Sort by pTM descending, then apply min_ptm and top-k. Returns (kept, dropped_count)."""
    if not candidates:
        return [], 0

    def ptm_key(c: dict) -> float:
        p = c.get("ptm")
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return float("-inf")
        return float(p)

    sorted_c = sorted(candidates, key=ptm_key, reverse=True)
    dropped = 0
    if min_ptm is not None:
        kept = []
        for c in sorted_c:
            p = c.get("ptm")
            if p is None or (isinstance(p, float) and np.isnan(p)):
                dropped += 1
                continue
            if float(p) >= min_ptm:
                kept.append(c)
            else:
                dropped += 1
        sorted_c = kept
    if top_k_per_target is not None and top_k_per_target > 0:
        extra = max(0, len(sorted_c) - top_k_per_target)
        dropped += extra
        sorted_c = sorted_c[:top_k_per_target]

    for rank, c in enumerate(sorted_c, start=1):
        c["ptm_rank_in_target"] = rank

    return sorted_c, dropped


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_rf3_results(
    rf3_dir: str,
    output_tsv: str,
    output_stats: str,
    min_ptm: float | None = None,
    top_k_per_target: int | None = None,
    exclude_target_ids: frozenset[str] | None = None,
):
    predictions_dir = os.path.join(rf3_dir, "predictions")
    configs_dir = os.path.join(rf3_dir, "rf3_configs")

    if not os.path.isdir(predictions_dir):
        print(f"ERROR: predictions directory not found: {predictions_dir}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    errors: list[str] = []
    filter_dropped_total = 0
    excluded_folders = 0

    pred_folders = sorted([
        d for d in os.listdir(predictions_dir)
        if os.path.isdir(os.path.join(predictions_dir, d))
    ])

    print(f"Found {len(pred_folders)} prediction folders")
    if exclude_target_ids:
        print(f"Excluding target IDs (folder or prefix before '_'): {', '.join(sorted(exclude_target_ids))}")
    print(
        f"pTM selection: sort descending, then "
        f"min_ptm={min_ptm}, top_k_per_target={top_k_per_target}"
    )

    for folder_name in pred_folders:
        base_id = folder_name.split("_")[0]
        if exclude_target_ids and (
            folder_name in exclude_target_ids or base_id in exclude_target_ids
        ):
            excluded_folders += 1
            continue

        target_id = base_id
        folder_path = os.path.join(predictions_dir, folder_name)

        meta, ref_ca, cdr_regions, ctx_err = _collect_target_context(configs_dir, target_id)
        if ctx_err:
            errors.append(ctx_err)
            continue
        assert meta is not None

        seed_dirs = sorted([
            d for d in os.listdir(folder_path)
            if d.startswith("seed") and os.path.isdir(os.path.join(folder_path, d))
        ])

        candidates: list[dict] = []
        for seed_dir_name in seed_dirs:
            sample_id = seed_dir_name
            seed_dir = os.path.join(folder_path, seed_dir_name)

            conf_files = [f for f in os.listdir(seed_dir) if f.endswith("summary_confidences.json")]
            if not conf_files:
                errors.append(f"{target_id}/{sample_id}: no summary_confidences.json")
                continue

            with open(os.path.join(seed_dir, conf_files[0])) as f:
                confidences = json.load(f)

            plddt = confidences.get("overall_plddt")
            ptm = confidences.get("ptm")

            cif_files = [f for f in os.listdir(seed_dir) if f.endswith("_model.cif")]
            if not cif_files:
                errors.append(f"{target_id}/{sample_id}: no model CIF file")
                continue

            cif_path = os.path.join(seed_dir, cif_files[0])
            candidates.append(
                {
                    "target_id": target_id,
                    "sample_id": sample_id,
                    "seed_dir": seed_dir,
                    "cif_path": cif_path,
                    "plddt": plddt,
                    "ptm": ptm,
                }
            )

        selected, n_drop = _filter_seeds_by_ptm(candidates, min_ptm, top_k_per_target)
        filter_dropped_total += n_drop

        for cand in selected:
            target_id = cand["target_id"]
            sample_id = cand["sample_id"]
            cif_path = cand["cif_path"]
            plddt = cand["plddt"]
            ptm = cand["ptm"]
            ptm_rank = cand.get("ptm_rank_in_target")

            try:
                pred_ca = parse_ca_from_cif(cif_path)
            except Exception as e:
                errors.append(f"{target_id}/{sample_id}: CIF parse error: {e}")
                continue

            try:
                ref_coords, pred_coords = paired_coords(ref_ca, pred_ca)
                global_crmsd = compute_crmsd(ref_coords, pred_coords)
                protein_tm = compute_tm_score(ref_coords, pred_coords, length_norm=len(ref_coords))
            except ValueError as e:
                errors.append(f"{target_id}/{sample_id}: global metrics error: {e}")
                continue

            row = {
                "target_id": target_id,
                "sample_id": sample_id,
                "ptm_rank_in_target": ptm_rank,
                "global_cRMSD": round(global_crmsd, 4),
                "protein_TM_score": round(protein_tm, 4),
                "plddt": round(plddt, 4) if plddt is not None else None,
                "pTM": round(ptm, 4) if ptm is not None else None,
            }

            for i, (chain, start, end) in enumerate(cdr_regions, 1):
                keys = region_keys(chain, start, end)
                try:
                    r, p = paired_coords(ref_ca, pred_ca, keys)
                    cdr_rmsd = compute_crmsd(r, p)
                    row[f"CDR{i}_cRMSD"] = round(cdr_rmsd, 4)
                except ValueError:
                    row[f"CDR{i}_cRMSD"] = None

            rows.append(row)

    if not rows:
        print("ERROR: no results extracted after pTM sort/filter", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    col_order = ["target_id", "sample_id", "ptm_rank_in_target", "global_cRMSD", "protein_TM_score"]
    cdr_cols = sorted([c for c in df.columns if c.startswith("CDR")])
    col_order += cdr_cols + ["plddt", "pTM"]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Results saved to: {output_tsv} ({len(df)} rows)")

    # --- Statistics (only on pTM-selected samples) ---
    metric_cols = ["global_cRMSD", "protein_TM_score"] + cdr_cols + ["plddt", "pTM"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    with open(output_stats, "w") as f:
        f.write("RF3 Prediction Results - Statistical Summary\n")
        f.write("(Metrics computed only after sorting seeds by pTM (desc) and applying filters.)\n")
        f.write("=" * 50 + "\n\n")
        if exclude_target_ids:
            f.write(
                f"Excluded target IDs: {', '.join(sorted(exclude_target_ids))}\n"
                f"Prediction folders skipped (excluded): {excluded_folders}\n\n"
            )
        f.write(f"Selection: min_ptm={min_ptm}, top_k_per_target={top_k_per_target}\n")
        f.write(f"Seeds dropped by pTM filter (all targets): {filter_dropped_total}\n\n")
        f.write(f"Total samples (after filter): {len(df)}\n")
        f.write(f"Total targets: {df['target_id'].nunique()}\n\n")

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

        f.write("\n\n--- Per-target mean statistics ---\n\n")
        target_means = df.groupby("target_id")[metric_cols].mean()
        f.write(f"{'Metric':<20s} {'Mean':>10s} {'Std':>10s} {'Median':>10s} {'Min':>10s} {'Max':>10s}\n")
        f.write("-" * 70 + "\n")
        for col in metric_cols:
            vals = target_means[col].dropna()
            if len(vals) == 0:
                continue
            f.write(
                f"{col:<20s} {vals.mean():>10.4f} {vals.std():>10.4f} "
                f"{vals.median():>10.4f} {vals.min():>10.4f} {vals.max():>10.4f}\n"
            )

    print(f"Statistics saved to: {output_stats}")

    if errors:
        print(f"\nWarnings/errors ({len(errors)}):")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract RF3 results: sort seeds by pTM (desc), optionally filter, then "
            "cRMSD (global + CDR) and whole-protein TM-score vs template."
        )
    )
    parser.add_argument(
        "--rf3_dir",
        type=str,
        required=True,
        help="Path to RF3 generations directory (containing predictions/ and rf3_configs/)",
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        default=None,
        help="Output TSV path (default: {rf3_dir}/rf3_results.tsv)",
    )
    parser.add_argument(
        "--output_stats",
        type=str,
        default=None,
        help="Output statistics TXT path (default: {rf3_dir}/rf3_statistics.txt)",
    )
    parser.add_argument(
        "--min-ptm",
        type=float,
        default=None,
        metavar="P",
        help="Keep only seeds with pTM >= P (after sorting by pTM descending).",
    )
    parser.add_argument(
        "--top-k-per-target",
        type=int,
        default=None,
        metavar="K",
        help="After filtering by --min-ptm, keep at most K seeds per target (best pTM first).",
    )
    parser.add_argument(
        "--exclude-protein-ids",
        type=str,
        default="",
        metavar="IDS",
        help="Comma or space separated IDs: skip matching prediction folders entirely "
        "(folder name or leading token before '_', e.g. 7pklL,8jbhE).",
    )
    args = parser.parse_args()

    output_tsv = args.output_tsv or os.path.join(args.rf3_dir, "rf3_results.tsv")
    output_stats = args.output_stats or os.path.join(args.rf3_dir, "rf3_statistics.txt")

    excl = parse_exclude_protein_ids(args.exclude_protein_ids)

    extract_rf3_results(
        rf3_dir=args.rf3_dir,
        output_tsv=output_tsv,
        output_stats=output_stats,
        min_ptm=args.min_ptm,
        top_k_per_target=args.top_k_per_target,
        exclude_target_ids=excl if excl else None,
    )


if __name__ == "__main__":
    main()
