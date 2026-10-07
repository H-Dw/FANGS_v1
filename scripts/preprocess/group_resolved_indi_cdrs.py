#!/usr/bin/env python3
"""Reproduce the selected annotations and identical-CDR groups in the INDI archive.

The calculation is extracted from notebook/connector_comparison.ipynb, cells
1-3 and 6-7 (zero-based indices). This command-line entry point was added during
provenance documentation; the notebook is the historical calculation source.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def reconstruct(annotations_path, manifest_path):
    annotations = pd.read_csv(annotations_path, sep="\t")
    manifest = pd.read_csv(manifest_path, sep="\t")
    selected = pd.merge(
        manifest, annotations, how="left", left_on="pdb_id", right_on="PDBChain"
    ).drop(columns=["pdb_id"])
    selected["PDB_ID"] = selected["PDBChain"].apply(lambda value: value[:4].upper())

    representatives = selected.drop_duplicates(subset=["PDB_ID"], keep="first")
    cdr_columns = ["CDR1", "CDR2", "CDR3"]
    members = representatives.groupby(cdr_columns).agg(
        Member=("PDBChain", lambda values: ",".join(values))
    ).reset_index()
    members = members.loc[
        members["Member"].apply(lambda value: len(value.split(",")) >= 2)
    ]
    groups = pd.merge(
        representatives.drop_duplicates(subset=cdr_columns), members, on=cdr_columns
    )
    return selected, representatives, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/INDI_database/INDI_info_full_cdr_info.tsv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/INDI_database/dupl_INDI_passed_cleaned.tsv"))
    parser.add_argument("--selected-out", type=Path, default=Path("data/INDI_database/analysis/INDI_info_selected_cdr_full_info.tsv"))
    parser.add_argument("--groups-out", type=Path, default=Path("data/INDI_database/analysis/cdr_grouped_output.tsv"))
    parser.add_argument("--verify-existing", action="store_true", help="Compare calculated values and row order with the existing output tables without writing files.")
    args = parser.parse_args()
    selected, representatives, groups = reconstruct(args.annotations, args.manifest)
    for table, path in [(selected, args.selected_out), (groups, args.groups_out)]:
        if args.verify_existing:
            existing = pd.read_csv(path, sep="\t")
            pd.testing.assert_frame_equal(table.reset_index(drop=True), existing.reset_index(drop=True), check_dtype=False)
        else:
            if not path.parent.is_dir():
                raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
            table.to_csv(path, sep="\t", index=False)
    print(json.dumps({
        "selected_chains": len(selected),
        "pdb_accession_representatives": len(representatives),
        "identical_cdr_groups": len(groups),
        "candidate_pairs": sum(len(value.split(",")) * (len(value.split(",")) - 1) // 2 for value in groups["Member"]),
        "existing_tables_match": True if args.verify_existing else None,
        "selected_output": str(args.selected_out),
        "groups_output": str(args.groups_out),
    }, indent=2))


if __name__ == "__main__":
    main()
