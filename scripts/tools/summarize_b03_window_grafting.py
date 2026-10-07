#!/usr/bin/env python3
"""Select the best B03 graft within the native-connector distance decile."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def windows(sequence: str) -> list[str]:
    return [sequence[index:index + 4] for index in range(len(sequence) - 3)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--carrier-table", type=Path, required=True)
    parser.add_argument("--carrier-pdb", type=Path, required=True)
    parser.add_argument("--candidate-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(args.scores_dir / "original_raw_embeddings.tsv", sep="\t")
    grafted = pd.read_csv(args.scores_dir / "grafted_raw_embeddings.tsv", sep="\t")
    for stage, frame in [("original", original), ("grafted", grafted)]:
        calculated = frame[[f"{stage}_euclidean_connector{index}_raw_embeddings" for index in [1, 2]]].mean(axis=1)
        if not np.allclose(calculated, frame.selected_euclidean_average, atol=1e-12):
            raise ValueError(f"Selected score differs from the two-connector mean: {stage}")
    ranking = original[["PDB_ID", "selected_euclidean_average"]].rename(
        columns={"selected_euclidean_average": "original_connector_score"}
    ).merge(
        grafted[["PDB_ID", "selected_euclidean_average"]].rename(
            columns={"selected_euclidean_average": "grafting_connector_score"}
        ), on="PDB_ID", validate="one_to_one",
    )
    ranking = ranking.sort_values(["original_connector_score", "PDB_ID"]).reset_index(drop=True)
    ranking.insert(0, "original_connector_rank", np.arange(1, len(ranking) + 1))
    threshold = float(ranking.original_connector_score.quantile(0.10))
    ranking["original_top10pct"] = ranking.original_connector_score <= threshold
    short = ranking[ranking.original_top10pct].sort_values(["grafting_connector_score", "PDB_ID"]).reset_index(drop=True)
    short.insert(0, "grafting_rank_within_original_top10pct", np.arange(1, len(short) + 1))
    archived_path = args.run_dir / "filter_best_selected_scores/top10pc_filtered_raw_embeddings.tsv"
    archived = pd.read_csv(archived_path, sep="\t")
    if short.PDB_ID.tolist() != archived.PDB_ID.tolist():
        raise ValueError("Shortlist identifier order differs from the archived decile selection")
    for new, old in [("original_connector_score", "selected_euclidean_average_original"),
                     ("grafting_connector_score", "selected_euclidean_average_grafted")]:
        if not np.allclose(short[new].round(2), archived[old], atol=1e-12):
            raise ValueError(f"Shortlist score differs from the archive: {new}")
    carrier = pd.read_csv(args.carrier_table, sep="\t").iloc[0]
    candidates = pd.read_csv(args.candidate_table, sep="\t", keep_default_na=False)
    expected_pairs = set(itertools.product(windows(carrier.CDR1), windows(carrier.CDR2)))
    if set(zip(candidates.CDR1, candidates.CDR2)) != expected_pairs or len(candidates) != 40:
        raise ValueError("Candidate table must enumerate all 8 × 5 windows")
    expected_ids = {"8v13HL_" + Path(name).stem for name in candidates.PDBChain}
    if set(ranking.PDB_ID) != expected_ids:
        raise ValueError("Ranked candidate identifiers differ from the window enumeration")
    structure = PDBParser(QUIET=True).get_structure("carrier", args.carrier_pdb)
    chain = structure[0]["A"]
    residues = [residue for residue in chain if residue.id[0] == " "]
    atom_sequence = "".join(seq1(residue.resname) for residue in residues)
    if atom_sequence != carrier.Sequence:
        raise ValueError("Carrier annotation differs from the PDB sequence")
    window_rows = []
    for cdr, replacement in [("CDR1", "GDYD"), ("CDR2", "WDNN")]:
        sequence = carrier[cdr]
        if atom_sequence.count(sequence) != 1:
            raise ValueError(f"Carrier CDR must map uniquely to its sequence: {cdr}")
        base = atom_sequence.index(sequence)
        for index, sequence_window in enumerate(windows(sequence)):
            ids = [residue.id for residue in residues[base + index:base + index + 4]]
            position_labels = [str(number) + insertion.strip() for _, number, insertion in ids]
            window_rows.append({"cdr": cdr, "window_index": index + 1, "window": sequence_window,
                                "pdb_start": position_labels[0], "pdb_end": position_labels[-1],
                                "pdb_positions": ",".join(position_labels), "replacement": replacement})
    window_table = pd.DataFrame(window_rows)
    generation = pd.read_csv(args.run_dir / "temp_generation/all_info/all_generation.tsv", sep="\t")
    if len(generation) != 2000 or generation.PDB_ID.nunique() != 40:
        raise ValueError("Generation archive must contain 2,000 samples from 40 candidates")
    best = short.iloc[0]
    selected_id = best.PDB_ID.removeprefix("8v13HL_")
    sequence_set = set(generation[generation.PDB_ID == selected_id].Sequence)
    if len(sequence_set) != 1:
        raise ValueError("Selected window pair has multiple generated sequences")
    sequence = sequence_set.pop()
    pair_row = candidates[candidates.PDBChain.map(lambda value: Path(value).stem) == selected_id].iloc[0]
    first_start = carrier.Sequence.index(carrier.CDR1) + carrier.CDR1.index(pair_row.CDR1)
    second_start = carrier.Sequence.index(carrier.CDR2) + carrier.CDR2.index(pair_row.CDR2)
    expected_sequence = (carrier.Sequence[:first_start] + "GDYD" + carrier.Sequence[first_start + 4:second_start]
                         + "WDNN" + carrier.Sequence[second_start + 4:])
    if sequence != expected_sequence:
        raise ValueError("Selected graft sequence differs from the two specified replacements")
    ranking.to_csv(args.output_dir / "B03_window_pair_ranking.tsv", sep="\t", index=False)
    short.to_csv(args.output_dir / "B03_top10pct_candidates.tsv", sep="\t", index=False)
    short.iloc[:1].to_csv(args.output_dir / "B03_final_candidate.tsv", sep="\t", index=False)
    window_table.to_csv(args.output_dir / "B03_window_definitions.tsv", sep="\t", index=False)
    (args.output_dir / "B03_final_candidate.fasta").write_text(f">{best.PDB_ID}\n{sequence}\n")
    filtered = pd.read_csv(args.run_dir / "extract_distance/filter/grafted_raw_embeddings.tsv", sep="\t")
    report = {"window_length": 4, "window_step": 1, "CDR1_windows": 8, "CDR2_windows": 5,
              "candidate_pairs": 40, "samples_per_candidate": 50, "generated_samples": 2000,
              "connector_filtered_samples": len(filtered), "original_top10pct_threshold": threshold,
              "original_top10pct_candidates": len(short), "selected_candidate": best.PDB_ID,
              "original_connector_rank": int(best.original_connector_rank),
              "original_connector_score": float(best.original_connector_score),
              "grafting_connector_score": float(best.grafting_connector_score),
              "CDR1_window": pair_row.CDR1, "CDR2_window": pair_row.CDR2,
              "sequence": sequence, "archived_shortlist_sha256": hashlib.sha256(archived_path.read_bytes()).hexdigest(),
              "selection_rule": "Minimum grafting connector score among the original connector distance decile"}
    (args.output_dir / "B03_selection_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
