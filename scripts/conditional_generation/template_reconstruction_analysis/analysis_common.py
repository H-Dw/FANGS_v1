"""Shared geometry and pairing utilities for the observed-chain benchmark."""

import csv
import json
from pathlib import Path

import gemmi
import numpy as np

MODELS = ("ESM3-Template", "AF3-Template")
REGIONS = (("Global", "global_CA_RMSD"), ("CDR1", "CDR1_CA_RMSD"),
           ("CDR2", "CDR2_CA_RMSD"), ("CDR3", "CDR3_CA_RMSD"))
CDRS = tuple(metric for region, metric in REGIONS if region != "Global")
COLORS = {MODELS[0]: "#789DB7", MODELS[1]: "#85AE98"}
INK = "#26343B"
CANONICAL = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_chain(path=None, text=None):
    if text is not None:
        structure = gemmi.make_structure_from_block(gemmi.cif.read_string(text).sole_block())
    else:
        structure = gemmi.read_structure(str(path))
    if not len(structure):
        raise ValueError("Empty structure")
    chains = []
    for chain in structure[0]:
        seq, xyz, residue_ids = [], [], set()
        for residue in chain:
            description = gemmi.find_tabulated_residue(residue.name)
            if not description.is_amino_acid():
                continue
            if residue.name not in CANONICAL:
                raise ValueError(f"Noncanonical protein residue {residue.name}")
            residue_id = str(residue.seqid)
            if residue_id in residue_ids:
                raise ValueError(f"Duplicate residue identifier {residue_id}")
            residue_ids.add(residue_id)
            seq.append(description.one_letter_code.upper())
            ca = [atom for atom in residue if atom.name == "CA"]
            atom = max(ca, key=lambda item: item.occ) if ca else None
            coord = None if atom is None else np.array((atom.pos.x, atom.pos.y, atom.pos.z))
            if coord is not None and not np.all(np.isfinite(coord)):
                raise ValueError("Non-finite C-alpha coordinate")
            xyz.append(coord)
        if seq:
            chains.append(("".join(seq), xyz))
    if len(chains) != 1:
        raise ValueError(f"Expected one protein chain; found {len(chains)}")
    return chains[0]


def rmsd(reference, prediction):
    a, b = np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("Need at least three corresponding C-alpha atoms")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Non-finite RMSD input")
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return float(np.sqrt(np.mean(np.sum((a - b @ u @ correction @ vt) ** 2, axis=1))))


def coordinate_difference(a, b):
    if len(a) != len(b) or any((x is None) != (y is None) for x, y in zip(a, b)):
        raise ValueError("C-alpha presence differs")
    return max((float(np.linalg.norm(x - y)) for x, y in zip(a, b) if x is not None), default=0.)


def cdr_mappings(config_root):
    result = {}
    for row in read_tsv(Path(config_root) / "cdr_mapping_report.tsv"):
        group = result.setdefault(row["PDBChain"], {})
        if row["CDR"] in group:
            raise ValueError(f"Duplicate mapping: {row['PDBChain']} {row['CDR']}")
        group[row["CDR"]] = row
    return result


def reference_info(config_root, target, mappings):
    config_path = Path(config_root) / "configs" / f"{target}_config.json"
    config = json.loads(config_path.read_text())
    if len(config["sequences"]) != 1 or "protein" not in config["sequences"][0]:
        raise ValueError("Benchmark requires one protein")
    protein = config["sequences"][0]["protein"]
    if len(protein["templates"]) != 1:
        raise ValueError("Benchmark requires one CDR template")
    template = protein["templates"][0]
    path = Path(template["mmcifPath"])
    seq, xyz = read_chain(path)
    if seq != protein["sequence"]:
        raise ValueError("Observed template and query sequences differ")
    region_map = mappings.get(target, {})
    if set(region_map) != {"CDR1", "CDR2", "CDR3"}:
        raise ValueError("Incomplete CDR mapping")
    indices = {}
    for region, row in region_map.items():
        qlo, qhi = map(int, row["query_indices"].split("-"))
        tlo, thi = map(int, row["template_indices"].split("-"))
        if (qlo, qhi) != (tlo, thi) or not 0 <= tlo <= thi < len(seq):
            raise ValueError(f"{region}: invalid observed-chain indices")
        if seq[tlo:thi + 1] != row["sequence"]:
            raise ValueError(f"{region}: CDR sequence mismatch")
        indices[region] = list(range(tlo, thi + 1))
        if len(indices[region]) < 3 or any(xyz[i] is None for i in indices[region]):
            raise ValueError(f"{region}: incomplete reference C-alpha coordinates")
    union = sorted(i for region in ("CDR1", "CDR2", "CDR3") for i in indices[region])
    if len(union) != len(set(union)):
        raise ValueError("Overlapping CDR indices")
    if template["queryIndices"] != union or template["templateIndices"] != union:
        raise ValueError("Config template indices differ from the three CDRs")
    return config, protein, path, seq, xyz, indices


def paired_rows(rows, expected_count=None):
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row["target"], {})
        if row["model"] not in MODELS or row["model"] in pair:
            raise ValueError("Duplicate or unexpected model")
        if not all(np.isfinite(float(row[metric])) for _, metric in REGIONS):
            raise ValueError("All four RMSDs must be finite")
        pair[row["model"]] = row
    if not pairs or any(set(pair) != set(MODELS) for pair in pairs.values()):
        raise ValueError("Incomplete paired cohort")
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError(f"Expected {expected_count} pairs; obtained {len(pairs)}")
    return dict(sorted(pairs.items()))


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)),
                median=float(np.median(a)), q1=float(np.quantile(a, .25)),
                q3=float(np.quantile(a, .75)), minimum=float(a.min()), maximum=float(a.max()))


def holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, prior = [0.] * len(p_values), 0.
    for rank, index in enumerate(order):
        prior = max(prior, min(1., (len(p_values) - rank) * p_values[index]))
        adjusted[index] = prior
    return adjusted


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
