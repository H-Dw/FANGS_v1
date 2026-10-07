#!/usr/bin/env python3
"""Validate generated AF3 jobs and compare CDR template positions with legacy jobs."""

import argparse
import csv
import json
import math
from pathlib import Path

import gemmi
from alphafold3.common import folding_input


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def template_residues(path):
    structure = gemmi.read_structure(str(path))
    chains = []
    for chain in structure[0]:
        residues = [r for r in chain
                    if r.entity_type == gemmi.EntityType.Polymer and
                    gemmi.find_tabulated_residue(r.name).one_letter_code != "?"]
        if residues:
            chains.append(residues)
    if len(chains) != 1:
        raise ValueError(f"Expected one protein chain in {path}")
    return chains[0]


def template_sequence(residues):
    return "".join(gemmi.find_tabulated_residue(r.name).one_letter_code
                   for r in residues).upper()


def ca_position(residue):
    ca = next((a for a in residue if a.name == "CA"), None)
    if ca is None:
        raise ValueError(f"No CA at residue {residue.seqid}")
    return (ca.pos.x, ca.pos.y, ca.pos.z)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config_root", type=Path, required=True)
    p.add_argument("--legacy_config_dir", type=Path, action="append", default=[])
    args = p.parse_args()
    root = args.config_root
    manifest = read_tsv(root / "generation_manifest.tsv")
    results = []
    errors = []
    for row in manifest:
        target = row["PDBChain"]
        config_path = Path(row["config"])
        try:
            parsed = list(folding_input.load_fold_inputs_from_path(config_path))
            if len(parsed) != 1:
                raise ValueError(f"AF3 parsed {len(parsed)} jobs")
            cfg = json.loads(config_path.read_text())
            protein = cfg["sequences"][0]["protein"]
            template = protein["templates"][0]
            sequence = protein["sequence"]
            qi, ti = template["queryIndices"], template["templateIndices"]
            if qi != ti or len(qi) != len(set(qi)):
                raise ValueError("Query/template indices not identical and unique")
            if any(i < 0 or i >= len(sequence) for i in qi):
                raise ValueError("CDR index out of sequence bounds")
            if len(sequence) != int(row["observed_sequence_length"]):
                raise ValueError("AF3 query length differs from source manifest")
            if Path(template["mmcifPath"]) != Path(row["template_cif"]):
                raise ValueError("Template path differs from source manifest")
            old = next((r / f"{target}_config.json"
                        for r in args.legacy_config_dir
                        if (r / f"{target}_config.json").exists()), None)
            old_template = (json.loads(old.read_text())["sequences"][0]["protein"]
                            ["templates"][0]) if old else None
            old_ti = old_template["templateIndices"] if old_template else []
            old_cfg = json.loads(old.read_text()) if old else None
            old_res = template_residues(Path(old_template["mmcifPath"])) if old else []
            new_res = template_residues(Path(template["mmcifPath"]))
            old_seq = template_sequence(old_res) if old else ""
            new_seq = template_sequence(new_res)
            if new_seq != sequence:
                raise ValueError("New template sequence differs from AF3 query")
            if old and old_seq != sequence:
                raise ValueError("AF3 query differs from ESM3/legacy reference sequence")
            if old and cfg["modelSeeds"] != old_cfg["modelSeeds"]:
                raise ValueError("Model seed differs from previous AF3 config")
            maximum_cdr_ca_shift = 0.0
            if old:
                for i in ti:
                    a, b = ca_position(new_res[i]), ca_position(old_res[i])
                    maximum_cdr_ca_shift = max(maximum_cdr_ca_shift,
                                               math.dist(a, b))
            results.append({
                "target": target,
                "AF3_parser_ok": True,
                "query_length": len(sequence),
                "CDR_index_count": len(qi),
                "query_equals_template_indices": qi == ti,
                "legacy_template_index_count": len(old_ti),
                "CDR_template_indices_equal_legacy": ti == old_ti if old else "",
                "new_query_equals_legacy_reference_sequence": sequence == old_seq,
                "seed_equals_legacy": cfg["modelSeeds"] == old_cfg["modelSeeds"] if old else "",
                "maximum_CDR_CA_shift_A": maximum_cdr_ca_shift,
                "legacy_config": str(old) if old else "",
            })
        except Exception as e:
            errors.append((target, str(e)))
    write_tsv(root / "af3_config_validation.tsv", results)
    print("AF3 parser passed", len(results), "of", len(manifest))
    print("CDR template indices identical to legacy", sum(
        r["CDR_template_indices_equal_legacy"] is True for r in results))
    print("CDR template indices changed", [r["target"] for r in results
          if r["CDR_template_indices_equal_legacy"] is False])
    print("Max CDR CA shift vs legacy (A)", max(
        r["maximum_CDR_CA_shift_A"] for r in results))
    print("errors", errors[:20])
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
