#!/usr/bin/env python3
"""Build AF3 CDR-template jobs using the coordinate-observed PDB chain sequence.

The INDI table supplies target identity and CDR strings only.  The AF3 query
sequence comes from the same resolved PDB chain converted into its mmCIF
template.  This mirrors the target-sequence choice in ESM3-Template generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import gemmi

from generate_af3_template_config import (
    _repair_mmcif_sequence_metadata,
    chain_sequence,
    choose_chain,
    polymer_residues,
)


CDRS = ("CDR1", "CDR2", "CDR3")
REQUIRED = {"PDBChain", "Sequence", *CDRS}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty TSV: {path}")
    return rows


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def select_rows(args: argparse.Namespace) -> list[tuple[dict[str, str], str]]:
    primary_rows = read_tsv(args.tsv_path)
    if not REQUIRED.issubset(primary_rows[0]):
        raise ValueError(f"Primary TSV needs {sorted(REQUIRED)}")
    primary: dict[str, dict[str, str]] = {}
    for row in primary_rows:
        target = row["PDBChain"].strip()
        if target in primary:
            raise ValueError(f"Duplicate target in primary TSV: {target}")
        primary[target] = row
    if args.targets_tsv:
        target_rows = read_tsv(args.targets_tsv)
        if "target" not in target_rows[0]:
            raise ValueError("--targets_tsv needs a 'target' column")
        targets = [r["target"].strip() for r in target_rows]
        if len(targets) != len(set(targets)):
            raise ValueError("Duplicate target in --targets_tsv")
    else:
        targets = list(primary)
    if args.limit is not None:
        targets = targets[:args.limit]
    fallback: dict[str, list[dict[str, str]]] = {}
    if args.fallback_tsv:
        for row in read_tsv(args.fallback_tsv):
            fallback.setdefault(row["PDBChain"].strip(), []).append(row)
    reference_cdr: dict[str, dict[str, str]] = {}
    if args.cdr_reference_report:
        for row in read_tsv(args.cdr_reference_report):
            reference_cdr.setdefault(row["PDBChain"].strip(), {})[
                row["CDR"]] = row["sequence"].strip().upper()
    selected = []
    for target in targets:
        if target in primary:
            selected.append((primary[target], args.tsv_path.name))
            continue
        candidates = fallback.get(target, [])
        if len(candidates) > 1 and target in reference_cdr:
            expected = reference_cdr[target]
            candidates = [r for r in candidates if all(
                r.get(name, "").strip().upper() == expected.get(name)
                for name in CDRS)]
        if len(candidates) != 1:
            raise ValueError(f"{target}: cannot resolve unique INDI row "
                             f"(remaining candidates={len(candidates)})")
        selected.append((candidates[0], args.fallback_tsv.name))
    return selected


def source_chain_info(pdb_path: Path, target: str):
    structure = gemmi.read_structure(str(pdb_path))
    if len(structure) == 0:
        raise ValueError("PDB has no model")
    chain = choose_chain(structure, target)
    all_residues = polymer_residues(chain)
    # ESM3's ProteinChain.from_pdb and the previous AF3 templates exclude
    # modified/noncanonical residues such as MSE, PCA, CRO and MLY.  Gemmi's
    # one-letter-code mapping would silently include them as ordinary amino
    # acids and shift all downstream CDR indices, so require canonical names.
    residues = [r for r in all_residues if r.name in STANDARD_RESIDUES]
    excluded = [f"{r.name}:{r.seqid.num}{r.seqid.icode.strip()}"
                for r in all_residues if r.name not in STANDARD_RESIDUES]
    sequence = chain_sequence(residues).upper()
    if not sequence or any(aa not in STANDARD_AA for aa in sequence):
        bad = sorted(set(sequence) - STANDARD_AA)
        raise ValueError(f"No protein sequence or unsupported residues: {bad}")
    return chain.name, residues, sequence, excluded


def convert_observed_chain_to_mmcif(residues: list[gemmi.Residue],
                                    output_path: Path) -> None:
    clean = gemmi.Structure()
    clean.name = output_path.stem
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for residue in residues:
        chain.add_residue(residue.clone())
    model.add_chain(chain)
    clean.add_model(model)
    clean.setup_entities()
    clean.make_mmcif_document().write_file(str(output_path))
    _repair_mmcif_sequence_metadata(output_path, residues)


def locate_cdrs(row: dict[str, str], residues: list[gemmi.Residue], sequence: str):
    query_indices: list[int] = []
    report: list[dict] = []
    cursor = 0
    for name in CDRS:
        cdr = row.get(name, "").strip().upper()
        if not cdr:
            raise ValueError(f"{name} missing in INDI row")
        candidates = []
        pos = sequence.find(cdr, cursor)
        while pos >= 0:
            candidates.append(pos)
            pos = sequence.find(cdr, pos + 1)
        if not candidates:
            raise ValueError(f"{name}={cdr!r} absent after prior CDR in observed chain")
        start = candidates[0]  # Same first-sequence-hit rule as the ESM3 script.
        end = start + len(cdr)
        for i in range(start, end):
            if not any(atom.name == "CA" for atom in residues[i]):
                raise ValueError(f"{name} residue {i + 1} has no C-alpha coordinate")
        if query_indices and start <= query_indices[-1]:
            raise ValueError(f"{name} overlaps the preceding CDR")
        query_indices.extend(range(start, end))
        report.append({
            "CDR": name,
            "sequence": cdr,
            "query_indices": f"{start}-{end - 1}",
            "template_indices": f"{start}-{end - 1}",
            "ordered_hits": len(candidates),
        })
        cursor = end
    return query_indices, report


def old_seed(target: str, roots: list[Path], default: int) -> tuple[int, str]:
    for root in roots:
        path = root / f"{target}_config.json"
        if path.exists():
            seeds = json.loads(path.read_text()).get("modelSeeds", [])
            if len(seeds) != 1:
                raise ValueError(f"{path}: expected one model seed")
            return int(seeds[0]), str(path)
    return default, "new_seed"


def validate_template(path: Path, expected_sequence: str,
                      indices: list[int]) -> None:
    structure = gemmi.read_structure(str(path))
    chains = [polymer_residues(c) for c in structure[0]
              if polymer_residues(c)]
    if len(chains) != 1:
        raise ValueError(f"Converted template has {len(chains)} polymer chains")
    residues = chains[0]
    if chain_sequence(residues).upper() != expected_sequence:
        raise ValueError("Converted template sequence differs from observed PDB chain")
    if any(not any(atom.name == "CA" for atom in residues[i]) for i in indices):
        raise ValueError("Converted CDR template has missing C-alpha coordinates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv_path", type=Path, required=True,
                        help="Primary clean INDI table")
    parser.add_argument("--pdb_dir", type=Path, required=True,
                        help="Directory with <PDBChain>.pdb coordinate files")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--targets_tsv", type=Path,
                        help="Optional cohort TSV with a target column")
    parser.add_argument("--fallback_tsv", type=Path,
                        help="INDI rows for targets absent from the primary table")
    parser.add_argument("--cdr_reference_report", type=Path,
                        help="Resolve duplicate fallback rows by prior CDR identity")
    parser.add_argument("--seed_config_dir", type=Path, action="append", default=[],
                        help="Reuse each target's previous modelSeed; repeat as needed")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--max_residues", type=int,
                        help="Skip longer resolved chains, recording them in skipped.tsv")
    parser.add_argument("--limit", type=int, help="First N requested targets for a pilot")
    args = parser.parse_args()
    if args.num_gpus < 1:
        parser.error("--num_gpus must be positive")
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory is not empty: {out}")
    selected = select_rows(args)
    out.mkdir(parents=True, exist_ok=True)
    config_dir = out / "configs"
    template_dir = out / "templates"
    shard_dir = out / "shards"
    config_dir.mkdir()
    template_dir.mkdir()
    for gpu in range(args.num_gpus):
        (shard_dir / f"gpu{gpu}").mkdir(parents=True)
    manifest, cdr_rows, skipped = [], [], []
    jobs = []
    for index, (row, indi_source) in enumerate(selected):
        target = row["PDBChain"].strip()
        if not re.fullmatch(r"[A-Za-z0-9]+", target):
            skipped.append({"PDBChain": target, "reason": "Invalid target ID"})
            continue
        pdb_path = args.pdb_dir.resolve() / f"{target}.pdb"
        if not pdb_path.is_file():
            skipped.append({"PDBChain": target, "reason": "Source PDB missing"})
            continue
        cif_path = template_dir / f"{target}.cif"
        config_path = config_dir / f"{target}_config.json"
        try:
            chain_id, residues, sequence, excluded_modified = source_chain_info(
                pdb_path, target)
            if args.max_residues and len(sequence) > args.max_residues:
                raise ValueError(f"Observed chain length {len(sequence)} exceeds "
                                 f"--max_residues {args.max_residues}")
            indices, mapped_cdrs = locate_cdrs(row, residues, sequence)
            convert_observed_chain_to_mmcif(residues, cif_path)
            validate_template(cif_path, sequence, indices)
            seed, seed_source = old_seed(target, args.seed_config_dir,
                                         args.seed + index)
            config = {
                "name": f"{target}_cdr_template_observed",
                "modelSeeds": [seed],
                "sequences": [{"protein": {
                    "id": "A", "sequence": sequence,
                    "unpairedMsa": "", "pairedMsa": "",
                    "templates": [{
                        "mmcifPath": str(cif_path.resolve()),
                        "queryIndices": indices,
                        "templateIndices": indices,
                    }],
                }}],
                "dialect": "alphafold3", "version": 4,
            }
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            for item in mapped_cdrs:
                cdr_rows.append({"config": config_path.name,
                                 "PDBChain": target, **item})
            manifest.append({
                "PDBChain": target,
                "INDI_source": indi_source,
                "source_pdb": str(pdb_path),
                "source_chain": chain_id,
                "observed_sequence_length": len(sequence),
                "INDI_sequence_length": len(row["Sequence"].strip()),
                "INDI_sequence_equals_observed":
                    row["Sequence"].strip().upper() == sequence,
                "excluded_modified_residues": ",".join(excluded_modified),
                "selected_CDR_residue_count": len(indices),
                "ambiguous_CDR_regions": ",".join(
                    item["CDR"] for item in mapped_cdrs
                    if item["ordered_hits"] > 1),
                "model_seed": seed,
                "seed_source": seed_source,
                "config": str(config_path),
                "template_cif": str(cif_path),
            })
            jobs.append((target, len(sequence), config_path))
        except Exception as exc:
            config_path.unlink(missing_ok=True)
            cif_path.unlink(missing_ok=True)
            skipped.append({"PDBChain": target, "reason": str(exc)})
    # Long sequences dominate AF3 memory and runtime.  Balance estimated
    # quadratic sequence cost across independent GPU workers.
    workloads = [0] * args.num_gpus
    for target, length, config_path in sorted(jobs,
                                              key=lambda x: (-x[1], x[0])):
        worker = min(range(args.num_gpus), key=lambda i: (workloads[i], i))
        shutil.copy2(config_path, shard_dir / f"gpu{worker}" / config_path.name)
        workloads[worker] += length * length
    write_tsv(out / "generation_manifest.tsv", manifest,
              ["PDBChain", "INDI_source", "source_pdb", "source_chain",
               "observed_sequence_length", "INDI_sequence_length",
               "INDI_sequence_equals_observed", "excluded_modified_residues",
               "selected_CDR_residue_count",
               "ambiguous_CDR_regions", "model_seed", "seed_source",
               "config", "template_cif"])
    write_tsv(out / "cdr_mapping_report.tsv", cdr_rows,
              ["config", "PDBChain", "CDR", "sequence", "query_indices",
               "template_indices", "ordered_hits"])
    write_tsv(out / "skipped.tsv", skipped, ["PDBChain", "reason"])
    summary = {
        "requested": len(selected),
        "generated": len(manifest),
        "skipped": len(skipped),
        "max_observed_chain_length": max((x[1] for x in jobs), default=0),
        "over_512_residues": sum(x[1] > 512 for x in jobs),
        "ambiguous_CDR_targets": sum(bool(r["ambiguous_CDR_regions"])
                                      for r in manifest),
        "targets_with_excluded_modified_residues": sum(
            bool(r["excluded_modified_residues"]) for r in manifest),
        "shard_estimated_cost": workloads,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if skipped:
        raise SystemExit(f"{len(skipped)} target(s) skipped; inspect {out / 'skipped.tsv'}")


if __name__ == "__main__":
    main()
