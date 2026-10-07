#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import csv
import itertools   # NEW


def generate_candidates_from_cdr(cdr_seq: str, motif_len: int):
    cdr_seq = cdr_seq.strip()
    if motif_len <= 0:
        return []
    if motif_len > len(cdr_seq):
        return []
    return [cdr_seq[i:i+motif_len] for i in range(len(cdr_seq) - motif_len + 1)]


def safe_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def link_pdb_for_candidate(pdb_file: str, out_dir: str, cand_name: str):
    """
    Create a hard link to pdb_file in out_dir, named <base>_<cand_name>.pdb.
    cand_name may encode several CDR fragment combinations at once,
    for example "CDR1-GGSE_CDR3-RRT".
    """
    safe_mkdir(out_dir)

    base = os.path.basename(pdb_file)
    root, ext = os.path.splitext(base)
    new_name = f"{root}_{cand_name}{ext}"
    target_path = os.path.join(out_dir, new_name)

    if not os.path.exists(target_path):
        os.link(pdb_file, target_path)

    return new_name


def process_file(
    input_tsv: str,
    output_tsv: str,
    motif: str,
    target_cdr: str,        # could be "CDR1,CDR2"
    none_token: str,
    output_mode: str,
    pdb_file: str = None,
    cand_pdb_dir: str = None
):

    motif_len = len(motif)

    # NEW: parse multiple CDRs
    target_cdr_list = [x.strip().upper() for x in target_cdr.split(",")]
    for t in target_cdr_list:
        if t not in ("CDR1", "CDR2", "CDR3"):
            raise ValueError(f"Invalid CDR name: {t}")

    # Load TSV
    rows = []
    with open(input_tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        req = {"Sequence", "CDR1", "CDR2", "CDR3", "PDBChain"}
        if not req.issubset(r.fieldnames):
            raise ValueError("TSV missing required fields")
        rows = list(r)

    out_fields = ["Sequence", "CDR1", "CDR2", "CDR3", "PDBChain"]
    with open(output_tsv, "w", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()

        for r in rows:

            seq = r["Sequence"].strip()
            pdbchain = r["PDBChain"].strip()

            # Collect candidates for each CDR
            cdr_candidate_map = {}   # { "CDR1": [...], "CDR3": [...] }

            for cdr in target_cdr_list:
                cdr_seq = r[cdr].strip()
                if not cdr_seq:
                    print(f"Warning: empty {cdr} for {seq}", file=sys.stderr)
                    break

                cands = generate_candidates_from_cdr(cdr_seq, motif_len)
                if not cands:
                    print(f"Warning: no candidates for {cdr} in {seq}", file=sys.stderr)
                    break

                cdr_candidate_map[cdr] = cands

            if len(cdr_candidate_map) != len(target_cdr_list):
                continue  # skip row

            # --- NEW: create full enumeration of combinations ---
            # itertools.product on candidate list values
            cdr_names = list(cdr_candidate_map.keys())
            candidate_lists = [cdr_candidate_map[c] for c in cdr_names]

            for comb in itertools.product(*candidate_lists):
                # comb is a tuple like ("GGSE", "RRST")
                comb_map = dict(zip(cdr_names, comb))

                # Create candidate name string: CDR1-GGSE_CDR3-RRST
                comb_str = "_".join([f"{cdr}-{fragment}" for cdr, fragment in comb_map.items()])

                # Link PDB file
                if pdb_file and cand_pdb_dir:
                    renamed_pdb = link_pdb_for_candidate(pdb_file, cand_pdb_dir, comb_str)
                else:
                    renamed_pdb = pdbchain

                # Output row
                if output_mode == "cleaned":
                    out_row = {
                        "Sequence": seq,
                        "CDR1": none_token,
                        "CDR2": none_token,
                        "CDR3": none_token,
                        "PDBChain": renamed_pdb
                    }
                else:
                    out_row = {
                        "Sequence": seq,
                        "CDR1": r["CDR1"],
                        "CDR2": r["CDR2"],
                        "CDR3": r["CDR3"],
                        "PDBChain": renamed_pdb
                    }

                # Overwrite target CDRs with generated fragments
                for cdr in comb_map:
                    out_row[cdr] = comb_map[cdr]

                writer.writerow(out_row)

    print(f"TSV written to: {output_tsv}")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-CDR candidates and link PDBs.")
    parser.add_argument("input_tsv")
    parser.add_argument("motif")
    parser.add_argument("target_cdr", help="Comma separated: e.g., CDR1,CDR3")
    parser.add_argument("-o", "--output")
    parser.add_argument("--none-token", default="None")
    parser.add_argument("--mode", default="original", choices=["cleaned", "original"])
    parser.add_argument("--pdb")
    parser.add_argument("--pdb_dir")

    args = parser.parse_args()

    out_path = args.output or (
        f"{os.path.splitext(os.path.basename(args.input_tsv))[0]}."
        f"{args.target_cdr.replace(',', '_')}_{len(args.motif)}_candidates.tsv"
    )

    process_file(
        input_tsv=args.input_tsv,
        output_tsv=out_path,
        motif=args.motif,
        target_cdr=args.target_cdr,
        none_token=args.none_token,
        output_mode=args.mode,
        pdb_file=args.pdb,
        cand_pdb_dir=args.pdb_dir
    )


if __name__ == "__main__":
    main()
