"""
Generate RF3 JSON configuration files for CDR-templated folding.

Reads a TSV file containing CDR1/CDR2/CDR3 sequences and PDBChain identifiers,
locates corresponding PDB template files, matches CDR sequences to residue
positions, and produces RF3-compatible JSON configs where CDR regions are fixed
as templates while framework regions fold freely.

Handles PDB files with empty chain identifiers (common when mmCIF structures
with multi-character auth_asym_ids are converted to PDB format) by writing
corrected copies with chain ID "A".

Usage:
    python generate_rf3_template_config.py \
        --tsv_path /path/to/INDI_info_full_cdr_info_dedup.tsv \
        --pdb_dir /path/to/pdb_renum/ \
        --output_dir /path/to/output/ \
        [--batch]
"""

import os
import json
import argparse
import warnings
from typing import Optional

import pandas as pd
import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure import filter_amino_acids
from biotite.sequence import ProteinSequence


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}

FALLBACK_CHAIN_ID = "A"
_COORD_RECORDS = {"ATOM  ", "HETATM", "TER   ", "ANISOU"}


def sanitize_pdb(pdb_path: str, output_dir: str) -> tuple[str, list[str]]:
    """
    Clean a PDB file to be compatible with RF3 / atomworks.

    Fixes applied (only when needed):
      1. Empty chain identifiers -> assign FALLBACK_CHAIN_ID at column 22.
      2. Multiple NMR MODELs -> keep only MODEL 1.
      3. Alternate conformations (altloc) -> keep only the highest-occupancy
         conformer (prefer altloc 'A' when tied), strip the altloc character.
      4. Strip HETATM / associated ANISOU records (water, ions, ligands) to
         prevent atomworks chain-renaming collisions when it separates polymer
         from non-polymer residues within the same chain.
      5. Insertion codes (e.g. 100, 100A, 100B from antibody numbering) ->
         renumber residues sequentially from 1, strip insertion codes.

    Returns:
        (path, fixes): path to the (possibly rewritten) PDB file and a list of
        fix descriptions applied (empty list if no changes were needed).
    """
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    has_empty_chain = False
    has_multi_model = False
    has_altloc = False
    has_hetatm = False
    has_insertion_code = False
    has_zero_occupancy = False

    model_count = 0
    occ_all_zero = True
    for line in lines:
        rec = line[:6]
        if rec == "MODEL ":
            model_count += 1
            if model_count > 1:
                has_multi_model = True
        if rec == "HETATM":
            has_hetatm = True
        if rec in _COORD_RECORDS and len(line) >= 27:
            if line[21] == " ":
                has_empty_chain = True
            if len(line) >= 17 and line[16] not in (" ", ""):
                has_altloc = True
            if line[26] not in (" ", ""):
                has_insertion_code = True
        if rec == "ATOM  " and len(line) >= 60:
            try:
                occ = float(line[54:60].strip())
                if occ != 0.0:
                    occ_all_zero = False
            except ValueError:
                occ_all_zero = False
    if occ_all_zero:
        has_zero_occupancy = True

    needs_fix = any([
        has_empty_chain, has_multi_model, has_altloc,
        has_hetatm, has_insertion_code, has_zero_occupancy,
    ])
    if not needs_fix:
        return pdb_path, []

    # --- Pass 1: filter lines (model, HETATM, altloc) ---
    filtered: list[str] = []
    in_model_1 = True
    current_model = 0

    for line in lines:
        rec = line[:6]

        if rec == "MODEL ":
            current_model += 1
            if current_model > 1:
                in_model_1 = False
            continue
        if rec == "ENDMDL":
            if not in_model_1:
                continue
            if has_multi_model:
                in_model_1 = False
            continue
        if not in_model_1 and rec in _COORD_RECORDS:
            continue

        if rec == "HETATM":
            continue
        if rec == "ANISOU" and len(line) >= 20:
            res_name = line[17:20].strip()
            if res_name not in THREE_TO_ONE:
                continue

        if rec == "ATOM  " and len(line) >= 17 and line[16] not in (" ", ""):
            if line[16] != "A":
                continue
            line = line[:16] + " " + line[17:]

        if rec in _COORD_RECORDS and len(line) >= 22 and line[21] == " ":
            line = line[:21] + FALLBACK_CHAIN_ID + line[22:]

        if has_zero_occupancy and rec == "ATOM  " and len(line) >= 60:
            line = line[:54] + "  1.00" + line[60:]

        filtered.append(line)

    # --- Pass 2: renumber residues sequentially (fixes insertion codes) ---
    out_lines: list[str] = []
    prev_reskey = None
    new_resid = 0

    for line in filtered:
        rec = line[:6]
        if rec in ("ATOM  ", "TER   ", "ANISOU") and len(line) >= 27:
            chain = line[21]
            orig_resid = line[22:26]
            icode = line[26]
            reskey = (chain, orig_resid, icode)

            if reskey != prev_reskey:
                new_resid += 1
                prev_reskey = reskey

            line = (
                line[:22]
                + f"{new_resid:>4d}"
                + " "          # clear insertion code (col 27)
                + line[27:]
            )

        out_lines.append(line)

    # --- Build fix descriptions ---
    fixes = []
    if has_empty_chain:
        fixes.append(f"empty chain -> '{FALLBACK_CHAIN_ID}'")
    if has_multi_model:
        fixes.append(f"multi-model ({model_count}) -> model 1 only")
    if has_altloc:
        fixes.append("altloc -> kept 'A' conformer")
    if has_hetatm:
        fixes.append("stripped HETATM")
    if has_insertion_code:
        fixes.append("insertion codes -> renumbered sequentially")
    if has_zero_occupancy:
        fixes.append("all-zero occupancy -> set to 1.00")

    fixed_dir = os.path.join(output_dir, "fixed_pdbs")
    os.makedirs(fixed_dir, exist_ok=True)
    fixed_path = os.path.join(fixed_dir, os.path.basename(pdb_path))

    with open(fixed_path, "w") as f:
        f.writelines(out_lines)

    return fixed_path, fixes


def extract_chain_info(pdb_path: str, chain_id: str):
    """
    Extract one-letter sequence and PDB residue IDs for a given chain.

    Returns:
        sequence (str): one-letter amino acid sequence
        res_ids (np.ndarray): PDB residue IDs corresponding to each residue
        chain_id (str): actual chain ID used (may differ from input if fallback was applied)
    """
    pdb_file = PDBFile.read(pdb_path)
    atom_array = pdb_file.get_structure(model=1)

    chain_mask = atom_array.chain_id == chain_id
    if not chain_mask.any():
        available = np.unique(atom_array.chain_id)
        if len(available) == 1:
            chain_id = available[0]
            chain_mask = atom_array.chain_id == chain_id
        else:
            raise ValueError(
                f"Chain '{chain_id}' not in {pdb_path}. Available: {available}"
            )

    chain_atoms = atom_array[chain_mask]

    ca_mask = chain_atoms.atom_name == "CA"
    aa_mask = filter_amino_acids(chain_atoms)
    mask = ca_mask & aa_mask
    ca_atoms = chain_atoms[mask]

    res_ids = ca_atoms.res_id
    res_names = ca_atoms.res_name

    sequence = ""
    for rn in res_names:
        sequence += THREE_TO_ONE.get(rn, "X")

    return sequence, res_ids, chain_id


def find_cdr_in_sequence(
    full_seq: str, cdr_seq: str
) -> Optional[tuple[int, int]]:
    """
    Locate a CDR subsequence within the full chain sequence.

    Returns:
        (start, end) as 0-based half-open indices, or None if not found.
    """
    if cdr_seq is None or (isinstance(cdr_seq, float) and pd.isna(cdr_seq)):
        return None
    cdr_seq = str(cdr_seq).strip()
    if cdr_seq == "":
        return None

    idx = full_seq.find(cdr_seq)
    if idx == -1:
        return None
    return (idx, idx + len(cdr_seq))


def build_template_selection(
    chain_id: str,
    full_seq: str,
    res_ids: np.ndarray,
    cdr_seqs: dict[str, str],
) -> tuple[list[str], list[dict]]:
    """
    Build AtomSelection strings for CDR regions.

    Args:
        chain_id: PDB chain identifier
        full_seq: full one-letter sequence of the chain
        res_ids: PDB residue IDs array
        cdr_seqs: dict mapping CDR name -> CDR amino acid sequence

    Returns:
        selections: list of AtomSelection query strings for template_selection
        cdr_info: list of dicts with CDR mapping details (for logging)
    """
    selections = []
    cdr_info = []

    for cdr_name in ["CDR1", "CDR2", "CDR3"]:
        cdr_seq = cdr_seqs.get(cdr_name)
        pos = find_cdr_in_sequence(full_seq, cdr_seq)
        if pos is None:
            continue

        start_idx, end_idx = pos
        pdb_res_start = int(res_ids[start_idx])
        pdb_res_end = int(res_ids[end_idx - 1])

        selection = f"{chain_id}/*/{pdb_res_start}-{pdb_res_end}"
        selections.append(selection)

        cdr_info.append({
            "cdr": cdr_name,
            "sequence": cdr_seq,
            "seq_idx": f"{start_idx}-{end_idx - 1}",
            "pdb_res": f"{pdb_res_start}-{pdb_res_end}",
            "selection": selection,
        })

    return selections, cdr_info


def generate_rf3_configs(
    tsv_path: str,
    pdb_dir: str,
    output_dir: str,
    batch: bool = False,
    n_splits: int = 1,
    predictions_dir: str | None = None,
):
    """
    Main entry: read TSV, match CDRs, write RF3 JSON configs.

    Args:
        tsv_path: path to TSV with columns [CDR1, CDR2, CDR3, PDBChain, ...]
        pdb_dir: directory containing {PDBChain}.pdb files
        output_dir: directory to write JSON config files
        batch: if True, also write a single batch JSON for all entries
        n_splits: split the batch config into N files for multi-GPU parallel runs
        predictions_dir: if set, skip entries whose predictions already exist
    """
    df = pd.read_csv(tsv_path, sep="\t")
    required_cols = {"CDR1", "CDR2", "CDR3", "PDBChain"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"TSV is missing required columns: {missing}")

    json_dir = os.path.join(output_dir, "rf3_configs")
    os.makedirs(json_dir, exist_ok=True)

    existing_ids: set[str] = set()
    if predictions_dir and os.path.isdir(predictions_dir):
        for entry in os.listdir(predictions_dir):
            ranking_csv = os.path.join(
                predictions_dir, entry, f"{entry}_ranking_scores.csv"
            )
            if os.path.isfile(ranking_csv):
                existing_ids.add(entry)
        print(f"Found {len(existing_ids)} completed predictions in {predictions_dir}")

    all_configs = []
    report_rows = []
    skipped = []

    fixed_count = 0

    for idx, row in df.iterrows():
        pdb_chain_str = str(row["PDBChain"]).strip()
        # PDB IDs are always 4 characters; the rest is the chain identifier
        pdb_id = pdb_chain_str[:4]
        original_chain_id = pdb_chain_str[4:]
        pdb_filename = f"{pdb_chain_str}.pdb"
        pdb_path = os.path.join(pdb_dir, pdb_filename)

        if not os.path.exists(pdb_path):
            skipped.append((pdb_chain_str, "PDB file not found"))
            continue

        example_id = f"{pdb_chain_str}_cdr_template"
        if example_id in existing_ids:
            skipped.append((pdb_chain_str, "prediction already exists"))
            continue

        # Sanitize: fix empty chain IDs, strip extra NMR models, resolve altloc
        pdb_path_for_rf3, fixes = sanitize_pdb(pdb_path, output_dir)
        was_fixed = len(fixes) > 0
        if was_fixed:
            fixed_count += 1
            print(f"  [FIX] {pdb_chain_str}: {'; '.join(fixes)}")

        has_empty_chain_fix = any("empty chain" in f for f in fixes)
        query_chain = (
            FALLBACK_CHAIN_ID if has_empty_chain_fix
            else original_chain_id[-1]
        )

        try:
            full_seq, res_ids, actual_chain = extract_chain_info(
                pdb_path_for_rf3, query_chain
            )
        except Exception as e:
            skipped.append((pdb_chain_str, f"parse error: {e}"))
            continue

        cdr_seqs = {
            "CDR1": row.get("CDR1"),
            "CDR2": row.get("CDR2"),
            "CDR3": row.get("CDR3"),
        }

        # Verify that at least one CDR can be found
        found_any = False
        missing_cdrs = []
        for cdr_name, cdr_seq in cdr_seqs.items():
            if cdr_seq is None or (isinstance(cdr_seq, float) and pd.isna(cdr_seq)):
                continue
            cdr_seq = str(cdr_seq).strip()
            if cdr_seq == "":
                continue
            pos = find_cdr_in_sequence(full_seq, cdr_seq)
            if pos is not None:
                found_any = True
            else:
                missing_cdrs.append(cdr_name)

        if not found_any:
            skipped.append((pdb_chain_str, f"no CDR matched in PDB sequence"))
            continue

        if missing_cdrs:
            warnings.warn(
                f"{pdb_chain_str}: CDRs not found in PDB sequence: "
                f"{missing_cdrs}"
            )

        selections, cdr_info = build_template_selection(
            actual_chain, full_seq, res_ids, cdr_seqs
        )

        if not selections:
            skipped.append((pdb_chain_str, "no valid selections generated"))
            continue

        config = {
            "name": f"{pdb_chain_str}_cdr_template",
            "components": [{"path": os.path.abspath(pdb_path_for_rf3)}],
            "template_selection": selections,
            "ground_truth_conformer_selection": selections,
        }

        all_configs.append(config)

        # Save individual config
        individual_path = os.path.join(json_dir, f"{pdb_chain_str}_config.json")
        with open(individual_path, "w") as f:
            json.dump([config], f, indent=4)

        for ci in cdr_info:
            report_rows.append({
                "PDBChain": pdb_chain_str,
                "CDR": ci["cdr"],
                "CDR_Sequence": ci["sequence"],
                "SeqIndex": ci["seq_idx"],
                "PDB_ResID": ci["pdb_res"],
                "AtomSelection": ci["selection"],
            })

    # Save batch config(s)
    if batch and all_configs:
        batch_path = os.path.join(json_dir, "batch_config.json")
        with open(batch_path, "w") as f:
            json.dump(all_configs, f, indent=4)
        print(f"Batch config saved to: {batch_path}")

        if n_splits > 1:
            chunk_size = (len(all_configs) + n_splits - 1) // n_splits
            for i in range(n_splits):
                chunk = all_configs[i * chunk_size : (i + 1) * chunk_size]
                if not chunk:
                    continue
                split_path = os.path.join(json_dir, f"batch_config_gpu{i}.json")
                with open(split_path, "w") as f:
                    json.dump(chunk, f, indent=4)
            print(f"Split into {n_splits} GPU configs: batch_config_gpu{{0..{n_splits-1}}}.json")

    # Save CDR mapping report
    if report_rows:
        report_df = pd.DataFrame(report_rows)
        report_path = os.path.join(output_dir, "cdr_mapping_report.tsv")
        report_df.to_csv(report_path, sep="\t", index=False)
        print(f"CDR mapping report saved to: {report_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RF3 Template Config Generation Summary")
    print(f"{'='*60}")
    print(f"Total entries in TSV:       {len(df)}")
    print(f"Configs generated:          {len(all_configs)}")
    print(f"PDBs sanitized:             {fixed_count}")
    print(f"Entries skipped:            {len(skipped)}")
    print(f"Individual configs dir:     {json_dir}")

    if skipped:
        print(f"\nSkipped entries:")
        for name, reason in skipped[:20]:
            print(f"  {name}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    return all_configs


def main():
    parser = argparse.ArgumentParser(
        description="Generate RF3 JSON configs for CDR-templated nanobody folding"
    )
    parser.add_argument(
        "--tsv_path",
        type=str,
        required=True,
        help="Path to TSV file with CDR info (requires CDR1/CDR2/CDR3/PDBChain columns)",
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        required=True,
        help="Directory containing PDB template files ({PDBChain}.pdb)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated JSON configs and reports",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Also generate a single batch JSON containing all configs",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=1,
        help="Split batch config into N files for multi-GPU parallel runs (requires --batch)",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default=None,
        help="Skip entries whose predictions already exist in this directory",
    )
    args = parser.parse_args()

    generate_rf3_configs(
        tsv_path=args.tsv_path,
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        batch=args.batch,
        n_splits=args.n_splits,
        predictions_dir=args.predictions_dir,
    )


if __name__ == "__main__":
    main()
