#!/usr/bin/env python3
"""Generate AlphaFold 3 JSON inputs with CDR structural templates.

Uses the RF3 TSV contract (Sequence, CDR1, CDR2, CDR3, PDBChain), converts
PDB templates to single-chain AF3-compatible mmCIF, maps CDR residues using
zero-based query/template indices, and disables MSA by writing empty fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

import gemmi


def polymer_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    result = []
    for residue in chain:
        if residue.entity_type != gemmi.EntityType.Polymer:
            continue
        code = gemmi.find_tabulated_residue(residue.name).one_letter_code
        if code and code != "?":
            result.append(residue)
    return result


def choose_chain(structure: gemmi.Structure, pdb_chain: str) -> gemmi.Chain:
    model = structure[0]
    requested = pdb_chain[4:] if len(pdb_chain) > 4 else pdb_chain
    candidates = list(model)
    for wanted in (requested, requested[-1:] if requested else "", "A"):
        if wanted:
            for chain in candidates:
                if chain.name == wanted and polymer_residues(chain):
                    return chain
    polymer_chains = [c for c in candidates if polymer_residues(c)]
    if len(polymer_chains) == 1:
        return polymer_chains[0]
    names = [c.name for c in polymer_chains]
    raise ValueError(f"cannot identify chain {requested!r}; available polymer chains: {names}")


def chain_sequence(residues: Iterable[gemmi.Residue]) -> str:
    return "".join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues)


def first_match(sequence: str, needle: str, start: int = 0) -> tuple[int, int] | None:
    needle = (needle or "").strip().upper()
    if not needle:
        return None
    pos = sequence.find(needle, start)
    return None if pos < 0 else (pos, pos + len(needle))


def _repair_mmcif_sequence_metadata(output_path: Path, residues: list[gemmi.Residue]):
    """Populate numeric polymer sequence IDs required by AF3's mmCIF parser.

    Gemmi can write valid coordinates for a newly assembled structure while
    leaving ``label_seq_id`` and the polymer sequence scheme unset.  AF3 uses
    those fields to map template residue indices, so fill them from the
    residue order used to build the normalized chain.
    """
    lines = output_path.read_text().splitlines()
    sequence = chain_sequence(residues)
    residue_keys = []
    for residue in residues:
        key = (residue.name, str(residue.seqid.num),
               residue.seqid.icode.strip() or '?')
        residue_keys.append(key)

    # Atom rows have the columns declared by the generated atom_site loop.
    # Replace label_seq_id with the normalized one-based polymer position.
    key_to_index = {key: i + 1 for i, key in enumerate(residue_keys)}
    repaired = []
    for line in lines:
        if line.startswith(('ATOM ', 'HETATM ')):
            fields = line.split()
            if len(fields) >= 19:
                key = (fields[5], fields[16], fields[9])
                # Gemmi writes '.' for an absent insertion code in atom_site.
                if key not in key_to_index:
                    key = (fields[5], fields[16], '?')
                if key in key_to_index:
                    fields[8] = str(key_to_index[key])
                    line = ' '.join(fields)
        repaired.append(line)

    # AF3 template featurisation requires a release date.  The source PDB
    # release date is not needed for the normalized coordinates, so use a
    # valid ISO date when the lightweight Gemmi writer omitted the header.
    if not any(line.startswith('_pdbx_audit_revision_history.revision_date')
               for line in repaired):
        entry_pos = next((i for i, line in enumerate(repaired)
                          if line.startswith('_entry.id ')), None)
        if entry_pos is None:
            raise ValueError('generated mmCIF has no entry header')
        repaired.insert(entry_pos + 1,
                        '_pdbx_audit_revision_history.revision_date 1970-01-01')

    # Add the sequence tables immediately before atom_type.  The normalized
    # chain has one entity (A) and one asym ID (the ID Gemmi wrote, Axp, etc.).
    try:
        atom_type_pos = next(i for i, line in enumerate(repaired)
                             if line == 'loop_' and i > 0
                             and repaired[i + 1] == '_atom_type.symbol')
    except StopIteration as exc:
        raise ValueError('generated mmCIF has no atom_type loop') from exc
    asym_id = next(
        line.split()[6] for line in repaired
        if line.startswith('ATOM ') and len(line.split()) >= 19
    )
    scheme = ['loop_', '_pdbx_poly_seq_scheme.asym_id',
              '_pdbx_poly_seq_scheme.entity_id', '_pdbx_poly_seq_scheme.seq_id',
              '_pdbx_poly_seq_scheme.mon_id', '_pdbx_poly_seq_scheme.pdb_seq_num',
              '_pdbx_poly_seq_scheme.auth_seq_num',
              '_pdbx_poly_seq_scheme.pdb_strand_id',
              '_pdbx_poly_seq_scheme.pdb_ins_code',
              '_pdbx_poly_seq_scheme.hetero']
    for i, residue in enumerate(residues, start=1):
        auth_num = str(residue.seqid.num)
        ins_code = residue.seqid.icode.strip() or '.'
        scheme.append(f'{asym_id} A {i} {residue.name} {auth_num} {auth_num} A {ins_code} n')
    scheme.extend([
        '', 'loop_', '_entity_poly_seq.entity_id', '_entity_poly_seq.num',
        '_entity_poly_seq.mon_id', '_entity_poly_seq.hetero',
    ])
    scheme.extend(f'A {i} {residue.name} n'
                  for i, residue in enumerate(residues, start=1))
    repaired[atom_type_pos:atom_type_pos] = scheme + ['']

    # Also provide the complete one-letter polymer sequence in the entity_poly
    # table instead of Gemmi's placeholder.
    for i, line in enumerate(repaired):
        if line == 'A polypeptide(L) A ?':
            repaired[i] = f'A polypeptide(L) A {sequence}'
            break
    output_path.write_text('\n'.join(repaired) + '\n')


def convert_chain_to_mmcif(pdb_path: Path, pdb_chain: str, output_path: Path):
    structure = gemmi.read_structure(str(pdb_path))
    if len(structure) == 0:
        raise ValueError("PDB has no model")
    source_chain = choose_chain(structure, pdb_chain)
    residues = polymer_residues(source_chain)
    if not residues:
        raise ValueError("selected chain has no polymer residues")
    clean = gemmi.Structure()
    clean.name = output_path.stem
    model = gemmi.Model("1")
    target_chain = gemmi.Chain("A")
    for residue in residues:
        target_chain.add_residue(residue.clone())
    model.add_chain(target_chain)
    clean.add_model(model)
    clean.setup_entities()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.make_mmcif_document().write_file(str(output_path))
    _repair_mmcif_sequence_metadata(output_path, residues)
    return chain_sequence(residues)


def build_config(row: dict[str, str], pdb_path: Path, mmcif_path: Path,
                 config_name: str, seed: int):
    sequence = row["Sequence"].strip().upper()
    template_sequence = convert_chain_to_mmcif(pdb_path, row["PDBChain"].strip(), mmcif_path)
    query_indices, template_indices, report = [], [], []
    q_cursor = t_cursor = 0
    for cdr_name in ("CDR1", "CDR2", "CDR3"):
        cdr = row.get(cdr_name, "").strip().upper()
        if not cdr:
            continue
        q_hit = first_match(sequence, cdr, q_cursor)
        t_hit = first_match(template_sequence, cdr, t_cursor)
        if q_hit is None or t_hit is None:
            raise ValueError(f"{cdr_name}={cdr!r} not found in query/template (query={q_hit}, template={t_hit})")
        q_start, q_end = q_hit
        t_start, t_end = t_hit
        if q_end - q_start != t_end - t_start:
            raise ValueError(f"{cdr_name} query/template lengths differ")
        query_indices.extend(range(q_start, q_end))
        template_indices.extend(range(t_start, t_end))
        report.append({
            "PDBChain": row["PDBChain"], "CDR": cdr_name, "sequence": cdr,
            "query_indices": f"{q_start}-{q_end - 1}",
            "template_indices": f"{t_start}-{t_end - 1}",
        })
        q_cursor, t_cursor = q_end, t_end
    if not query_indices:
        raise ValueError("no CDR mappings were generated")
    config = {
        "name": config_name,
        "modelSeeds": [int(seed)],
        "sequences": [{"protein": {
            "id": "A", "sequence": sequence,
            "unpairedMsa": "", "pairedMsa": "",
            "templates": [{
                "mmcifPath": str(mmcif_path.resolve()),
                "queryIndices": query_indices,
                "templateIndices": template_indices,
            }],
        }}],
        "dialect": "alphafold3", "version": 4,
    }
    return config, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv_path", required=True)
    p.add_argument("--pdb_dir", required=True,
                   help="Primary directory containing <PDBChain>.pdb files")
    p.add_argument("--fixed_pdb_dir", default=None,
                   help="Prefer an RF3-sanitized PDB from this directory when present")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if args.num_gpus < 1:
        raise ValueError("--num_gpus must be positive")
    out = Path(args.output_dir).resolve()
    config_dir, template_dir, shard_dir = out / "configs", out / "templates", out / "shards"
    config_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_gpus):
        (shard_dir / f"gpu{i}").mkdir(parents=True, exist_ok=True)
    with open(args.tsv_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    required = {"Sequence", "CDR1", "CDR2", "CDR3", "PDBChain"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"TSV must contain columns: {sorted(required)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    report_rows, skipped, generated = [], [], 0
    for idx, row in enumerate(rows):
        pdb_id = row["PDBChain"].strip()
        pdb_path = Path(args.pdb_dir).resolve() / f"{pdb_id}.pdb"
        if args.fixed_pdb_dir:
            fixed_path = Path(args.fixed_pdb_dir).resolve() / f"{pdb_id}.pdb"
            if fixed_path.exists():
                pdb_path = fixed_path
        if not pdb_path.exists():
            skipped.append((pdb_id, "PDB file not found")); continue
        config_name = f"{pdb_id}_cdr_template"
        mmcif_path = template_dir / f"{pdb_id}.cif"
        try:
            config, report = build_config(row, pdb_path, mmcif_path, config_name, args.seed + idx)
        except Exception as exc:
            skipped.append((pdb_id, str(exc))); continue
        config_path = config_dir / f"{pdb_id}_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy2(config_path, shard_dir / f"gpu{generated % args.num_gpus}" / config_path.name)
        for item in report:
            item["config"] = config_path.name
            report_rows.append(item)
        generated += 1
    with (out / "cdr_mapping_report.tsv").open("w", newline="") as f:
        fields = ["config", "PDBChain", "CDR", "sequence", "query_indices", "template_indices"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(report_rows)
    with (out / "skipped.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t"); writer.writerow(["PDBChain", "reason"]); writer.writerows(skipped)
    print(f"Generated {generated} AF3 configs in {config_dir}")
    print(f"GPU shards: {shard_dir}/gpu{{0..{args.num_gpus - 1}}}")
    print(f"Skipped {len(skipped)} entries; see {out / 'skipped.tsv'}")


if __name__ == "__main__":
    main()
