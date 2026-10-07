#!/usr/bin/env python3
"""Sequence-mapped regional RMSD analysis for PDB files.

Coordinates from structure 1 are compared directly with structure 2.  No
regional fitting is performed.  When a transform is supplied it is applied as
``rotation @ coordinate + translation`` before RMSD calculation.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
    # Common modified residues with an unambiguous parent amino acid.
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C",
    "CSD": "C", "CME": "C", "CSS": "C", "KCX": "K", "LLP": "K",
    "MLY": "K", "M3L": "K", "ALY": "K", "HYP": "P", "PCA": "E",
    "FME": "M", "NLE": "L", "ORN": "K", "SAR": "G",
}
BACKBONE_ATOMS = ("N", "CA", "C", "O")


class PDBParseError(ValueError):
    """Raised when an ATOM record cannot be represented safely."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.details}


@dataclass(frozen=True)
class ResidueID:
    chain: str
    resseq: int
    icode: str

    def as_dict(self) -> dict[str, Any]:
        return {"chain": self.chain, "resseq": self.resseq, "icode": self.icode}


@dataclass
class _AtomChoice:
    coordinate: np.ndarray
    altloc: str
    occupancy: float
    line_number: int


@dataclass
class Residue:
    residue_id: ResidueID
    resname: str
    one_letter: str
    atoms: dict[str, _AtomChoice] = field(default_factory=dict)

    def audit_dict(self) -> dict[str, Any]:
        return {
            **self.residue_id.as_dict(),
            "resname": self.resname,
            "one_letter": self.one_letter,
        }


@dataclass
class PDBStructure:
    path: str
    residues: list[Residue]

    @property
    def sequence(self) -> str:
        return "".join(residue.one_letter for residue in self.residues)

    def audit_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sequence": self.sequence,
            "residue_count": len(self.residues),
            "residues": [residue.audit_dict() for residue in self.residues],
        }


def _altloc_rank(choice: _AtomChoice) -> tuple[int, float, int]:
    """Blank wins over A, then other altlocs use highest occupancy.

    Occupancy breaks ties inside a priority class and earlier input order is the
    final deterministic tie-breaker.
    """
    priority = 2 if choice.altloc == "" else 1 if choice.altloc == "A" else 0
    return priority, choice.occupancy, -choice.line_number


def parse_pdb(path: str | Path) -> PDBStructure:
    """Parse ATOM records from the first PDB MODEL in file order."""
    pdb_path = Path(path)
    try:
        lines = pdb_path.read_text().splitlines()
    except OSError as exc:
        raise PDBParseError("pdb_read_error", str(exc), path=str(pdb_path)) from exc

    has_models = any(line[:6].strip() == "MODEL" for line in lines)
    in_first_model = not has_models
    first_model_started = False
    residues: "OrderedDict[ResidueID, Residue]" = OrderedDict()

    for line_number, line in enumerate(lines, start=1):
        record = line[:6].strip()
        if has_models and record == "MODEL":
            if first_model_started:
                break
            first_model_started = True
            in_first_model = True
            continue
        if has_models and in_first_model and record == "ENDMDL":
            break
        if record != "ATOM" or not in_first_model:
            continue
        if len(line) < 54:
            raise PDBParseError(
                "malformed_atom_record", "ATOM record is too short",
                path=str(pdb_path), line_number=line_number,
            )

        atom_name = line[12:16].strip()
        altloc = line[16:17].strip().upper()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip()
        icode = line[26:27].strip()
        try:
            resseq = int(line[22:26].strip())
            coordinate = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
            occupancy_text = line[54:60].strip() if len(line) >= 60 else ""
            occupancy = float(occupancy_text) if occupancy_text else 0.0
        except ValueError as exc:
            raise PDBParseError(
                "malformed_atom_record", "Invalid residue number, coordinate, or occupancy",
                path=str(pdb_path), line_number=line_number,
            ) from exc

        if resname not in THREE_TO_ONE:
            raise PDBParseError(
                "unknown_residue", f"Cannot map residue {resname!r} to one-letter code",
                path=str(pdb_path), line_number=line_number, resname=resname,
                residue_id=ResidueID(chain, resseq, icode).as_dict(),
            )

        residue_id = ResidueID(chain, resseq, icode)
        residue = residues.get(residue_id)
        if residue is None:
            residue = Residue(residue_id, resname, THREE_TO_ONE[resname])
            residues[residue_id] = residue
        elif residue.resname != resname:
            raise PDBParseError(
                "conflicting_residue_names",
                "A residue ID has multiple residue names in ATOM records",
                path=str(pdb_path), line_number=line_number,
                residue_id=residue_id.as_dict(), names=[residue.resname, resname],
            )

        candidate = _AtomChoice(coordinate, altloc, occupancy, line_number)
        current = residue.atoms.get(atom_name)
        if current is None or _altloc_rank(candidate) > _altloc_rank(current):
            residue.atoms[atom_name] = candidate

    if has_models and not first_model_started:
        raise PDBParseError("missing_first_model", "MODEL marker was not parseable", path=str(pdb_path))
    return PDBStructure(str(pdb_path), list(residues.values()))


def _failure(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message, **details}}


def _region_sequences(spec: Any) -> tuple[str, str]:
    if isinstance(spec, str):
        return spec, spec
    if isinstance(spec, Mapping):
        left = spec.get("structure1", spec.get("sequence1"))
        right = spec.get("structure2", spec.get("sequence2"))
        if isinstance(left, str) and isinstance(right, str):
            return left, right
    if isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)) and len(spec) == 2:
        left, right = spec
        if isinstance(left, str) and isinstance(right, str):
            return left, right
    raise ValueError("region specification must be a sequence string or a pair of strings")


def _all_occurrences(sequence: str, query: str) -> list[int]:
    starts: list[int] = []
    start = sequence.find(query)
    while start != -1:
        starts.append(start)
        start = sequence.find(query, start + 1)
    return starts


def _metric(
    residues1: Sequence[Residue],
    residues2: Sequence[Residue],
    atom_names: Iterable[str],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    atom_names = tuple(atom_names)
    missing: list[dict[str, Any]] = []
    coordinates1: list[np.ndarray] = []
    coordinates2: list[np.ndarray] = []
    for index, (residue1, residue2) in enumerate(zip(residues1, residues2)):
        for atom_name in atom_names:
            absent = []
            if atom_name not in residue1.atoms:
                absent.append("structure1")
            if atom_name not in residue2.atoms:
                absent.append("structure2")
            if absent:
                missing.append({
                    "region_offset": index,
                    "atom": atom_name,
                    "missing_in": absent,
                    "structure1_residue_id": residue1.residue_id.as_dict(),
                    "structure2_residue_id": residue2.residue_id.as_dict(),
                })
                continue
            coordinates1.append(residue1.atoms[atom_name].coordinate)
            coordinates2.append(residue2.atoms[atom_name].coordinate)
    if missing:
        return _failure(
            "missing_atoms", "Required atoms are missing", required_atoms=list(atom_names),
            missing=missing,
        )

    array1 = np.asarray(coordinates1, dtype=float)
    array2 = np.asarray(coordinates2, dtype=float)
    transformed1 = (rotation @ array1.T).T + translation
    squared_distances = np.sum((transformed1 - array2) ** 2, axis=1)
    return {
        "success": True,
        "rmsd": float(np.sqrt(np.mean(squared_distances))),
        "atom_count": int(len(squared_distances)),
    }


def _validate_transform(
    rotation: Any | None, translation: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_array = np.eye(3, dtype=float) if rotation is None else np.asarray(rotation, dtype=float)
    translation_array = np.zeros(3, dtype=float) if translation is None else np.asarray(translation, dtype=float)
    if rotation_array.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if translation_array.shape != (3,):
        raise ValueError("translation must have shape (3,)")
    if not np.all(np.isfinite(rotation_array)) or not np.all(np.isfinite(translation_array)):
        raise ValueError("rotation and translation must contain only finite numbers")
    return rotation_array, translation_array


def analyze_regions(
    structure1: str | Path,
    structure2: str | Path,
    regions: Mapping[str, Any],
    *,
    rotation: Any | None = None,
    translation: Any | None = None,
    transform: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze named sequence regions and return a JSON-serializable result.

    Region values may be one string (used on both sides), a two-string sequence,
    or a mapping containing ``structure1`` and ``structure2`` strings.
    """
    result: dict[str, Any] = {
        "success": False,
        "structure1": {"path": str(structure1)},
        "structure2": {"path": str(structure2)},
        "regions": {},
        "combined": _failure("regions_not_successful", "Combined metrics require every region to succeed"),
    }
    if transform is not None:
        if rotation is not None or translation is not None:
            result["error"] = {
                "code": "invalid_transform", "message": "Use transform or rotation/translation, not both",
            }
            return result
        rotation = transform.get("rotation")
        translation = transform.get("translation")
    try:
        rotation_array, translation_array = _validate_transform(rotation, translation)
        parsed1 = parse_pdb(structure1)
        parsed2 = parse_pdb(structure2)
    except PDBParseError as exc:
        result["error"] = exc.as_dict()
        return result
    except (TypeError, ValueError) as exc:
        result["error"] = {"code": "invalid_transform", "message": str(exc)}
        return result

    result["structure1"] = parsed1.audit_dict()
    result["structure2"] = parsed2.audit_dict()
    result["transform"] = {
        "rotation": rotation_array.tolist(), "translation": translation_array.tolist(),
        "convention": "rotation @ coordinate + translation",
    }

    combined_residues1: list[Residue] = []
    combined_residues2: list[Residue] = []
    all_regions_successful = bool(regions)
    for name, spec in regions.items():
        region_result: dict[str, Any] = {"success": False}
        result["regions"][str(name)] = region_result
        try:
            sequence1, sequence2 = _region_sequences(spec)
        except ValueError as exc:
            region_result["error"] = {"code": "invalid_region_spec", "message": str(exc)}
            all_regions_successful = False
            continue
        sequence1 = sequence1.strip().upper()
        sequence2 = sequence2.strip().upper()
        region_result["sequence1"] = sequence1
        region_result["sequence2"] = sequence2
        if not sequence1 or not sequence2:
            region_result["error"] = {"code": "empty_region", "message": "Region sequence must be non-empty"}
            all_regions_successful = False
            continue
        if sequence1 != sequence2:
            region_result["error"] = {
                "code": "region_sequence_mismatch",
                "message": "Region sequences for the two structures must be identical",
            }
            all_regions_successful = False
            continue

        starts1 = _all_occurrences(parsed1.sequence, sequence1)
        starts2 = _all_occurrences(parsed2.sequence, sequence2)
        if len(starts1) != 1 or len(starts2) != 1:
            code = "region_not_found" if not starts1 or not starts2 else "region_not_unique"
            region_result["error"] = {
                "code": code,
                "message": "Region must match exactly once in each structure sequence",
                "match_starts": {"structure1": starts1, "structure2": starts2},
                "match_counts": {"structure1": len(starts1), "structure2": len(starts2)},
            }
            all_regions_successful = False
            continue

        start1, start2 = starts1[0], starts2[0]
        residues1 = parsed1.residues[start1:start1 + len(sequence1)]
        residues2 = parsed2.residues[start2:start2 + len(sequence2)]
        region_result["mapping"] = [
            {
                "offset": offset,
                "one_letter": sequence1[offset],
                "structure1": residue1.audit_dict(),
                "structure2": residue2.audit_dict(),
            }
            for offset, (residue1, residue2) in enumerate(zip(residues1, residues2))
        ]
        region_result["ca"] = _metric(
            residues1, residues2, ("CA",), rotation_array, translation_array,
        )
        region_result["backbone"] = _metric(
            residues1, residues2, BACKBONE_ATOMS, rotation_array, translation_array,
        )
        region_result["success"] = bool(
            region_result["ca"]["success"] and region_result["backbone"]["success"]
        )
        if region_result["success"]:
            combined_residues1.extend(residues1)
            combined_residues2.extend(residues2)
        else:
            all_regions_successful = False

    if not regions:
        result["error"] = {"code": "no_regions", "message": "At least one region is required"}
    if all_regions_successful:
        result["combined"] = {
            "success": True,
            "region_names": [str(name) for name in regions],
            "ca": _metric(
                combined_residues1, combined_residues2, ("CA",),
                rotation_array, translation_array,
            ),
            "backbone": _metric(
                combined_residues1, combined_residues2, BACKBONE_ATOMS,
                rotation_array, translation_array,
            ),
        }
        result["success"] = True
    return result


def _parse_region_arguments(values: Sequence[str]) -> dict[str, str]:
    regions: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"region must be NAME=SEQ: {value!r}")
        name, sequence = value.split("=", 1)
        if not name:
            raise argparse.ArgumentTypeError("region name must be non-empty")
        if name in regions:
            raise argparse.ArgumentTypeError(f"duplicate region name: {name!r}")
        regions[name] = sequence
    return regions


def _load_transform(value: str) -> Mapping[str, Any]:
    candidate = Path(value)
    try:
        text = candidate.read_text() if candidate.is_file() else value
        transform = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid transform JSON: {exc}") from exc
    if not isinstance(transform, dict):
        raise argparse.ArgumentTypeError("transform JSON must be an object")
    return transform


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure1", required=True, help="Aligned or raw structure 1 PDB")
    parser.add_argument("--structure2", required=True, help="Reference structure 2 PDB")
    parser.add_argument("--region", action="append", required=True, metavar="NAME=SEQ")
    parser.add_argument(
        "--transform-json", type=_load_transform,
        help="JSON object or path containing rotation and translation",
    )
    parser.add_argument("--out-json", help="Write JSON output to this path instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        regions = _parse_region_arguments(args.region)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    result = analyze_regions(
        args.structure1, args.structure2, regions, transform=args.transform_json,
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).write_text(output + "\n")
    else:
        print(output)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
