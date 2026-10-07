#!/usr/bin/env python3
"""Export the VHH2 structural shortlist and its AbNatiV comparison sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


def read_fasta(path: Path) -> dict[str, str]:
    records = list(SeqIO.parse(path, "fasta"))
    result = {record.id: str(record.seq) for record in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate sequence identifiers: {path}")
    return result


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--abnativ-dir", type=Path, required=True)
    parser.add_argument("--rank-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.run_dir / "extract_distance/filter_best"
    original_col = "original_euclidean_all_raw_embeddings"
    grafted_col = "grafted_euclidean_all_raw_embeddings"
    original = pd.read_csv(base / "original_raw_embeddings.tsv", sep="\t")
    grafted = pd.read_csv(base / "grafted_raw_embeddings.tsv", sep="\t")
    rank = pd.read_csv(args.rank_table, sep="\t")
    expected = grafted[["PDB_ID", grafted_col]].merge(
        original[["PDB_ID", original_col]], on="PDB_ID", validate="one_to_one"
    )
    threshold = float(original[original_col].quantile(0.10))
    expected = expected[expected[original_col] <= threshold].sort_values(grafted_col)
    if rank["PDB_ID"].tolist() != expected["PDB_ID"].tolist():
        raise ValueError("Rank table differs from the native-distance decile shortlist")
    for column in [original_col, grafted_col]:
        if not np.allclose(rank[column], expected[column].round(2), atol=1e-12):
            raise ValueError(f"Ranked scores differ: {column}")
    rank.insert(0, "structural_rank", np.arange(1, len(rank) + 1))
    scores = pd.read_csv(args.abnativ_dir / "5m2jD_VH_abnativ_seq_scores.csv")
    vhh = pd.read_csv(args.abnativ_dir / "5m2jD_VHH_abnativ_seq_scores.csv")
    scores = scores[["seq_id", "input_seq", "AbNatiV VH Score"]].merge(
        vhh[["seq_id", "input_seq", "AbNatiV VHH Score"]],
        on=["seq_id", "input_seq"], validate="one_to_one",
    )
    fasta_path = args.abnativ_dir / "5m2jD_humanization_comparasion.fasta"
    fasta = read_fasta(fasta_path)
    if set(scores.seq_id) != set(fasta):
        raise ValueError("Score identifiers differ from the comparison FASTA")
    if any(fasta[row.seq_id] != row.input_seq for row in scores.itertuples()):
        raise ValueError("Scored sequences differ from the comparison FASTA")
    scores["PDB_ID"] = scores.seq_id.str.replace(r"^FANGS_", "", regex=True)
    ranked = rank.merge(scores, on="PDB_ID", validate="one_to_one")
    ranked["both_scores_pass"] = (
        (ranked["AbNatiV VH Score"] >= 0.8) & (ranked["AbNatiV VHH Score"] >= 0.8)
    )
    if len(ranked) != 15 or int(ranked.both_scores_pass.sum()) != 9:
        raise ValueError("VHH2 shortlist must contain 15 candidates and 9 dual-threshold passes")
    filtered = pd.read_csv(args.run_dir / "extract_distance/filter/grafted_raw_embeddings.tsv", sep="\t")
    representative = filtered.sort_values([grafted_col, "Filename"]).drop_duplicates("PDB_ID")
    ranked = ranked.merge(representative[["PDB_ID", "Filename"]], on="PDB_ID", validate="one_to_one")
    generation = pd.read_csv(args.run_dir / "temp_generation/all_info/all_generation.tsv", sep="\t")
    ranked = ranked.merge(generation[["Filename", "Sequence", "pTM", "cRMSD"]], on="Filename", validate="one_to_one")
    if not ranked.input_seq.eq(ranked.Sequence).all():
        raise ValueError("Ranked reconstruction sequences differ from AbNatiV inputs")
    ranked.drop(columns=["Sequence"]).to_csv(args.output_dir / "VHH2_ranked_candidates.tsv", sep="\t", index=False)
    native = original[["PDB_ID", original_col]].sort_values([original_col, "PDB_ID"])
    native.insert(0, "native_rank", np.arange(1, len(native) + 1))
    native.to_csv(args.output_dir / "VHH2_native_framework_ranking.tsv", sep="\t", index=False)
    selected_ids = ranked.seq_id.tolist()
    hudiff_ids = [name for name in fasta if name.startswith("HuDiff_")]
    reference_ids = [name for name in fasta if name.startswith(("Original_", "TNF30_"))]
    for filename, ids in [
        ("VHH2_fangs_top15.fasta", selected_ids),
        ("VHH2_hudiff_comparators.fasta", hudiff_ids),
        ("VHH2_comparison_24.fasta", reference_ids + hudiff_ids + selected_ids),
    ]:
        write_fasta(args.output_dir / filename, [(name, fasta[name]) for name in ids])
    counts = {
        "IGHV3_alleles": len(read_fasta(args.library_dir / "IGHV3_aa_alleles.fasta")),
        "CDR3_FR4_fusions": len(read_fasta(args.library_dir / "IGHV3_5M2J_CDR3_FR4.fasta")),
        "Kabat_numbered_sequences": pd.read_csv(args.library_dir / "IGHV3_5M2J_CDR3_FR4_kabat.csv").shape[1] - 1,
        "mutated_sequences": len(read_fasta(args.library_dir / "IGHV3_5M2J_CDR3_FR4_74A_94R.fasta")),
        "Boltz_carrier_structures": len(list((args.library_dir / "boltz2_output_IGHV3_5M2J_CDR3_FR4_74A_94R_extracted/pdb").glob("*.pdb"))),
        "prompt_records": len(pd.read_csv(args.run_dir / "temp_generation/temp_input.tsv", sep="\t")),
        "generated_carriers": int(generation.PDB_ID.nunique()),
        "generated_samples": len(generation),
        "connector_filtered_samples": len(filtered),
        "native_distance_10pct_quantile": threshold,
        "structurally_selected_candidates": len(ranked),
        "AbNatiV_dual_threshold_pass": int(ranked.both_scores_pass.sum()),
        "AbNatiV_dual_threshold_pass_fraction": float(ranked.both_scores_pass.mean()),
        "HuDiff_comparators": len(hudiff_ids),
        "reference_sequences": len(reference_ids),
        "full_comparison_fasta_sequences": len(fasta),
        "full_comparison_fasta_sha256": hashlib.sha256(fasta_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "VHH2_humanization_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
