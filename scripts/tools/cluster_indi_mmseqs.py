#!/usr/bin/env python3
"""
Cluster INDI nanobody entries by sequence identity (MMseqs2).

By default, MMseqs clustering uses the Sequence column of the CDR TSV.
PDB atom sequences are available through ``--sequence-source structure``.
An external FASTA is used only with ``--sequence-source fasta``; ``--fasta``
does not override the default TSV source. For each threshold, write:
  - representative ID list
  - raw cluster TSV (representative, member)
  - CDR info table for representatives (same columns as the input TSV)
  - representative structures mapped to the clustering target, plus a report
  - representative FASTA taken from those extracted structures (not original PDBs)

Example:
  python scripts/tools/cluster_indi_mmseqs.py \\
    --cdr-tsv data/INDI_database/INDI_info_full_cdr_info_dedup_clean.tsv \\
    --pdb-dir data/INDI_database/passed_pdbs \\
    --out-dir data/INDI_database/mmseqs_cluster_structure \\
    --identities 0.95 0.90 0.85 \\
    --mmseqs /data1/dhuang/MMseqs2-16-747c6/build/bin/mmseqs
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from Bio.Align import PairwiseAligner
from Bio.PDB import PDBIO, PDBParser, Select


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from readPDBSeq import (  # noqa: E402
    get_sequence_from_pdb,
    non_standard_three_to_one,
    three_to_one,
)


DEFAULT_IDENTITIES = (0.95, 0.90, 0.85)
CDR_COLUMNS = ("Sequence", "CDR1", "CDR2", "CDR3", "PDBChain", "Member")
EXTRACTION_COLUMNS = (
    "PDBChain", "structure_chain", "target_length", "structure_chain_length",
    "matched_residues", "missing_residues", "sequence_identity",
    "alignment_score", "missing_positions", "output_pdb",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster sequences in an INDI CDR table with MMseqs2 (Sequence column by default), "
            "and write a representative list, a CDR table, representative structures mapped "
            "back to the source PDB entries, and a representative-sequence FASTA derived from "
            "the extracted structures."
        )
    )
    parser.add_argument(
        "--cdr-tsv",
        type=Path,
        required=True,
        help="Input CDR table (for example, INDI_info_full_cdr_info_dedup_clean.tsv)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        required=True,
        help="Directory of structures named {PDBChain}.pdb (passed_pdbs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output root directory; each identity threshold is written to its own subdirectory",
    )
    parser.add_argument(
        "--identities",
        type=float,
        nargs="+",
        default=list(DEFAULT_IDENTITIES),
        help="Sequence-identity thresholds in [0, 1]; default: 0.95 0.90 0.85",
    )
    parser.add_argument(
        "--mmseqs",
        type=str,
        default="mmseqs",
        help="Path to the mmseqs executable (resolved from PATH by default)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads used by MMseqs2",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Alignment coverage threshold passed as -c (default, 0.8)",
    )
    parser.add_argument(
        "--cov-mode",
        type=int,
        default=1,
        help="Coverage mode passed as --cov-mode (default 1: require target coverage; suited to redundancy reduction)",
    )
    parser.add_argument(
        "--sequence-source",
        choices=("tsv", "structure", "fasta"),
        default="tsv",
        help=(
            "Sequence source for MMseqs2 clustering; tsv by default (Sequence column of the CDR table). "
            "structure parses sequences from PDB atoms; fasta reads the file given by --fasta"
        ),
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help=(
            "External FASTA. Used for clustering only with --sequence-source fasta; "
            "in structure mode it may be used as a cache of structure-derived sequences. "
            "Ignored when the default tsv source is selected, so that pdb2seq.fasta is not applied unintentionally"
        ),
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Retain the MMseqs2 temporary directory",
    )
    parser.add_argument(
        "--expand-members",
        action="store_true",
        help=(
            "Expand the Member column of the output CDR table to every original member assigned to the same cluster "
            "(by default, only PDBChain identifiers within the cluster are listed)"
        ),
    )
    return parser.parse_args(argv)


def identity_tag(identity: float) -> str:
    pct = int(round(identity * 100))
    return f"id{pct}"


def read_cdr_rows(tsv_path: Path) -> List[Dict[str, str]]:
    with tsv_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [c for c in CDR_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CDR TSV missing columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows in {tsv_path}")
    return rows


def parse_fasta(fasta_path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    seq_id: Optional[str] = None
    chunks: List[str] = []
    with fasta_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    sequences[seq_id] = "".join(chunks)
                seq_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if seq_id is not None:
            sequences[seq_id] = "".join(chunks)
    return sequences


def write_fasta(path: Path, records: Iterable[Tuple[str, str]]) -> int:
    count = 0
    with path.open("w") as handle:
        for seq_id, sequence in records:
            if not sequence:
                continue
            handle.write(f">{seq_id}\n{sequence}\n")
            count += 1
    return count


def _sequences_from_fasta(existing_fasta: Path, wanted: Sequence[str]) -> Dict[str, str]:
    all_seqs = parse_fasta(existing_fasta)
    missing = [cid for cid in wanted if cid not in all_seqs]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} PDBChain IDs missing from {existing_fasta}: {missing[:5]}"
        )
    return {cid: "".join(all_seqs[cid].split()).upper() for cid in wanted}


def _sequences_from_tsv(
    rows: Sequence[Dict[str, str]], wanted: Sequence[str],
) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    empty_seq = []
    for row, cid in zip(rows, wanted):
        sequence = "".join(row["Sequence"].split()).upper()
        if sequence:
            sequences[cid] = sequence
        else:
            empty_seq.append(cid)
    if empty_seq:
        raise ValueError(f"Empty TSV Sequence values for: {empty_seq[:5]}")
    return sequences


def _sequences_from_structures(wanted: Sequence[str], pdb_dir: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    missing_pdb: List[str] = []
    empty_seq: List[str] = []
    for cid in wanted:
        pdb_path = pdb_dir / f"{cid}.pdb"
        if not pdb_path.exists():
            missing_pdb.append(cid)
            continue
        sequence = get_sequence_from_pdb(str(pdb_path))
        if not sequence:
            empty_seq.append(cid)
            continue
        sequences[cid] = sequence
    if missing_pdb:
        raise FileNotFoundError(
            f"{len(missing_pdb)} PDB files missing under {pdb_dir}: {missing_pdb[:5]}"
        )
    if empty_seq:
        raise ValueError(f"Empty structure sequences for: {empty_seq[:5]}")
    return sequences


def build_clustering_fasta(
    rows: Sequence[Dict[str, str]],
    pdb_dir: Path,
    out_fasta: Path,
    existing_fasta: Optional[Path] = None,
    sequence_source: str = "tsv",
) -> Dict[str, str]:
    """Build the exact PDBChain-to-sequence collection passed to MMseqs."""
    wanted = [row["PDBChain"].strip() for row in rows]

    if sequence_source == "tsv":
        if existing_fasta is not None:
            print(
                f"[WARNING] Ignoring --fasta {existing_fasta}; "
                "MMseqs clustering uses the CDR TSV Sequence column "
                "(pass --sequence-source fasta to cluster from a FASTA file)"
            )
        sequences = _sequences_from_tsv(rows, wanted)
        source_label = "CDR TSV Sequence column"
    elif sequence_source == "fasta":
        if existing_fasta is None:
            raise ValueError("--sequence-source fasta requires --fasta")
        sequences = _sequences_from_fasta(existing_fasta, wanted)
        source_label = f"FASTA {existing_fasta}"
    elif sequence_source == "structure":
        if existing_fasta is not None:
            sequences = _sequences_from_fasta(existing_fasta, wanted)
            source_label = f"structure FASTA cache {existing_fasta}"
        else:
            sequences = _sequences_from_structures(wanted, pdb_dir)
            source_label = "PDB structure atoms"
    else:
        raise ValueError(f"Unknown sequence source: {sequence_source}")

    count = write_fasta(out_fasta, ((cid, sequences[cid]) for cid in wanted))
    print(
        f"[INFO] MMseqs clustering will use {count} sequences from "
        f"{source_label} -> {out_fasta}"
    )
    return sequences


def build_structure_fasta(
    rows: Sequence[Dict[str, str]],
    pdb_dir: Path,
    out_fasta: Path,
    existing_fasta: Optional[Path] = None,
) -> Dict[str, str]:
    """Backward-compatible helper for structure-derived clustering sequences."""
    return build_clustering_fasta(
        rows, pdb_dir, out_fasta, existing_fasta, sequence_source="structure"
    )


def _protein_residues(chain) -> Tuple[str, List[object]]:
    sequence: List[str] = []
    residues: List[object] = []
    residue_codes = {**three_to_one, **non_standard_three_to_one}
    for residue in chain:
        code = residue_codes.get(residue.get_resname().strip().upper())
        if code is not None:
            sequence.append(code)
            residues.append(residue)
    return "".join(sequence), residues


def _alignment_mapping(structure_sequence: str, target_sequence: str) -> Dict[str, object]:
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    # Extra structure residues at either end appear as end gaps in the query row.
    # Keeping target-row end gaps penalized forces the complete target sequence
    # to participate, while these query end-gap settings locate an internal region.
    aligner.query_left_open_gap_score = 0.0
    aligner.query_left_extend_gap_score = 0.0
    aligner.query_right_open_gap_score = 0.0
    aligner.query_right_extend_gap_score = 0.0
    best: Optional[Dict[str, object]] = None
    for alignment in aligner.align(structure_sequence, target_sequence):
        matched: Dict[int, int] = {}
        aligned_pairs = 0
        for (s_start, s_end), (t_start, t_end) in zip(
            alignment.aligned[0], alignment.aligned[1]
        ):
            block_length = min(s_end - s_start, t_end - t_start)
            for offset in range(block_length):
                s_index = int(s_start + offset)
                t_index = int(t_start + offset)
                aligned_pairs += 1
                if structure_sequence[s_index] == target_sequence[t_index]:
                    matched[t_index] = s_index
        identity = len(matched) / len(target_sequence) if target_sequence else 0.0
        key = (float(alignment.score), len(matched), identity, -min(matched, default=0))
        if best is None or key > best["key"]:
            best = {
                "score": float(alignment.score), "matched": matched,
                "aligned_pairs": aligned_pairs, "identity": identity, "key": key,
            }
    return best or {"score": 0.0, "matched": {}, "aligned_pairs": 0, "identity": 0.0}


class _MappedResidueSelect(Select):
    def __init__(self, model_id: int, chain_id: str, residues: Set[object]):
        self.model_id = model_id
        self.chain_id = chain_id
        self.residues = residues

    def accept_model(self, model):
        return model.id == self.model_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue in self.residues


def extract_representative_structure(
    pdb_chain: str, target_sequence: str, pdb_path: Path, output_pdb: Path,
) -> Dict[str, str]:
    """Map a clustering target onto the best first-model chain and save exact matches."""
    target_sequence = "".join(target_sequence.split()).upper()
    if not target_sequence:
        raise ValueError(f"Empty clustering target sequence for {pdb_chain}")
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file missing for representative {pdb_chain}: {pdb_path}")

    structure = PDBParser(QUIET=True).get_structure(pdb_chain, str(pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration as exc:
        raise ValueError(f"No model found in {pdb_path}") from exc

    candidates = []
    for chain_index, chain in enumerate(model):
        chain_sequence, residues = _protein_residues(chain)
        if not chain_sequence:
            continue
        mapping = _alignment_mapping(chain_sequence, target_sequence)
        matched = mapping["matched"]
        key = (
            float(mapping["score"]), len(matched), float(mapping["identity"]),
            int(mapping["aligned_pairs"]), -chain_index,
        )
        candidates.append((key, chain, chain_sequence, residues, mapping))

    if not candidates:
        raise ValueError(f"No protein chains found in first model of {pdb_path}")
    _, chain, chain_sequence, residues, mapping = max(candidates, key=lambda item: item[0])
    matched = mapping["matched"]
    if not matched:
        raise ValueError(
            f"No matching residues for representative {pdb_chain} target in {pdb_path}"
        )

    selected_residues = {residues[s_index] for s_index in matched.values()}
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb), _MappedResidueSelect(model.id, chain.id, selected_residues))

    missing = [
        f"{index + 1}:{letter}" for index, letter in enumerate(target_sequence)
        if index not in matched
    ]
    if missing:
        print(
            f"[WARNING] {pdb_chain}: structure chain {chain.id!r} is missing "
            f"{len(missing)}/{len(target_sequence)} target residues: {','.join(missing)}"
        )
    return {
        "PDBChain": pdb_chain, "structure_chain": chain.id,
        "target_length": str(len(target_sequence)),
        "structure_chain_length": str(len(chain_sequence)),
        "matched_residues": str(len(matched)),
        "missing_residues": str(len(missing)),
        "sequence_identity": f"{len(matched) / len(target_sequence):.6f}",
        "alignment_score": f"{float(mapping['score']):.3f}",
        "missing_positions": ",".join(missing), "output_pdb": str(output_pdb),
    }


def extract_representative_structures(
    reps: Sequence[str], clustering_sequences: Dict[str, str], pdb_dir: Path,
    output_dir: Path, report_path: Path,
) -> List[Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for rep in reps:
        if rep not in clustering_sequences:
            raise KeyError(f"No clustering sequence found for representative {rep}")
        report_rows.append(extract_representative_structure(
            rep, clustering_sequences[rep], pdb_dir / f"{rep}.pdb",
            output_dir / f"{rep}.pdb",
        ))
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=EXTRACTION_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(report_rows)
    return report_rows


def write_rep_seq_fasta_from_extracted_structures(
    reps: Sequence[str], extraction_dir: Path, out_fasta: Path,
) -> Dict[str, str]:
    """Write representative sequences parsed from extracted structures only."""
    sequences: Dict[str, str] = {}
    missing_pdb: List[str] = []
    empty_seq: List[str] = []
    for rep in reps:
        pdb_path = extraction_dir / f"{rep}.pdb"
        if not pdb_path.exists():
            missing_pdb.append(rep)
            continue
        sequence = get_sequence_from_pdb(str(pdb_path))
        if not sequence:
            empty_seq.append(rep)
            continue
        sequences[rep] = sequence
    if missing_pdb:
        raise FileNotFoundError(
            f"{len(missing_pdb)} extracted representative PDBs missing under "
            f"{extraction_dir}: {missing_pdb[:5]}"
        )
    if empty_seq:
        raise ValueError(
            f"Empty sequences from extracted representative structures: {empty_seq[:5]}"
        )
    count = write_fasta(out_fasta, ((rep, sequences[rep]) for rep in reps))
    print(
        f"[INFO] Wrote {count} representative sequences from extracted structures "
        f"-> {out_fasta}"
    )
    return sequences


def run_mmseqs_cluster(
    mmseqs: str,
    fasta: Path,
    out_prefix: Path,
    tmp_dir: Path,
    identity: float,
    coverage: float,
    cov_mode: int,
    threads: int,
) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        mmseqs,
        "easy-cluster",
        str(fasta),
        str(out_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(identity),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs cluster TSV not found: {cluster_tsv}")
    return cluster_tsv


def load_clusters(cluster_tsv: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Return ordered unique representatives and mapping rep -> members."""
    clusters: Dict[str, List[str]] = defaultdict(list)
    reps: List[str] = []
    seen_reps = set()
    with cluster_tsv.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            if rep not in seen_reps:
                seen_reps.add(rep)
                reps.append(rep)
            clusters[rep].append(member)
    return reps, dict(clusters)


def write_rep_list(path: Path, reps: Sequence[str]) -> None:
    with path.open("w") as handle:
        for rep in reps:
            handle.write(f"{rep}\n")


def build_cdr_table_for_reps(
    rows: Sequence[Dict[str, str]],
    reps: Sequence[str],
    clusters: Dict[str, List[str]],
    expand_members: bool,
) -> List[Dict[str, str]]:
    by_chain = {row["PDBChain"]: row for row in rows}
    missing = [rep for rep in reps if rep not in by_chain]
    if missing:
        raise KeyError(f"Cluster reps not found in CDR TSV: {missing[:5]}")

    out_rows: List[Dict[str, str]] = []
    for rep in reps:
        base = by_chain[rep]
        member_chains = clusters.get(rep, [rep])
        if expand_members:
            member_ids: List[str] = []
            seen = set()
            for chain in member_chains:
                for mid in by_chain[chain]["Member"].split(","):
                    mid = mid.strip()
                    if mid and mid not in seen:
                        seen.add(mid)
                        member_ids.append(mid)
            member_str = ",".join(member_ids)
        else:
            member_str = ",".join(member_chains)

        out_rows.append(
            {
                "Sequence": base["Sequence"],
                "CDR1": base["CDR1"],
                "CDR2": base["CDR2"],
                "CDR3": base["CDR3"],
                "PDBChain": rep,
                "Member": member_str,
            }
        )
    return out_rows


def write_cdr_tsv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(CDR_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in CDR_COLUMNS})


def write_summary(
    path: Path,
    identity: float,
    n_input: int,
    n_reps: int,
    clusters: Dict[str, List[str]],
) -> None:
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    with path.open("w") as handle:
        handle.write(f"min_seq_id\t{identity}\n")
        handle.write(f"n_input\t{n_input}\n")
        handle.write(f"n_clusters\t{n_reps}\n")
        handle.write(f"max_cluster_size\t{sizes[0] if sizes else 0}\n")
        handle.write(f"singleton_clusters\t{sum(1 for s in sizes if s == 1)}\n")


def resolve_mmseqs(mmseqs: str) -> str:
    path = shutil.which(mmseqs) if "/" not in mmseqs else mmseqs
    if path is None or not Path(path).exists():
        # allow absolute path even if not executable bit quirks
        candidate = Path(mmseqs)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(
            f"mmseqs not found: {mmseqs}. Pass --mmseqs /path/to/mmseqs"
        )
    return path


def main() -> None:
    args = parse_args()
    mmseqs = resolve_mmseqs(args.mmseqs)
    identities = args.identities
    for identity in identities:
        if not 0.0 < identity <= 1.0:
            raise ValueError(f"Identity must be in (0, 1], got {identity}")

    rows = read_cdr_rows(args.cdr_tsv)
    print(f"[INFO] Loaded {len(rows)} CDR rows from {args.cdr_tsv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clustering_fasta = args.out_dir / "clustering_sequences.fasta"
    print(f"[INFO] MMseqs sequence source: {args.sequence_source}")
    clustering_sequences = build_clustering_fasta(
        rows=rows,
        pdb_dir=args.pdb_dir,
        out_fasta=clustering_fasta,
        existing_fasta=args.fasta,
        sequence_source=args.sequence_source,
    )

    for identity in identities:
        tag = identity_tag(identity)
        id_dir = args.out_dir / tag
        id_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = id_dir / "cluster"
        tmp_dir = id_dir / "tmp"

        cluster_tsv = run_mmseqs_cluster(
            mmseqs=mmseqs,
            fasta=clustering_fasta,
            out_prefix=out_prefix,
            tmp_dir=tmp_dir,
            identity=identity,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
            threads=args.threads,
        )
        reps, clusters = load_clusters(cluster_tsv)

        rep_list_path = id_dir / "rep_list.txt"
        cdr_out_path = id_dir / "cdr_info.tsv"
        summary_path = id_dir / "summary.tsv"

        write_rep_list(rep_list_path, reps)
        cdr_rows = build_cdr_table_for_reps(
            rows=rows,
            reps=reps,
            clusters=clusters,
            expand_members=args.expand_members,
        )
        write_cdr_tsv(cdr_out_path, cdr_rows)
        write_summary(summary_path, identity, len(rows), len(reps), clusters)
        extraction_dir = id_dir / "cluster_rep_structures"
        extraction_report = id_dir / "cluster_rep_structure_extraction.tsv"
        extract_representative_structures(
            reps, clustering_sequences, args.pdb_dir, extraction_dir, extraction_report
        )
        # Overwrite MMseqs2 cluster_rep_seq.fasta: that file otherwise contains
        # clustering input sequences (TSV or full original PDB residues).
        rep_seq_fasta = Path(f"{out_prefix}_rep_seq.fasta")
        write_rep_seq_fasta_from_extracted_structures(
            reps, extraction_dir, rep_seq_fasta
        )

        if not args.keep_tmp and tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        print(
            f"[INFO] {tag}: {len(rows)} -> {len(reps)} reps | "
            f"rep_list={rep_list_path} | cdr={cdr_out_path} | "
            f"structures={extraction_dir} | rep_seq={rep_seq_fasta}"
        )

    print(f"[DONE] Results under {args.out_dir}")


if __name__ == "__main__":
    main()
