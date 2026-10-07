import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "tools" / "region_rmsd.py"
SPEC = importlib.util.spec_from_file_location("region_rmsd", MODULE_PATH)
region_rmsd = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = region_rmsd
assert SPEC.loader is not None
SPEC.loader.exec_module(region_rmsd)


def atom_line(serial, atom, resname, chain, resseq, xyz, *, icode="", altloc="", occupancy=1.0):
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:>4s}{altloc:1s}{resname:>3s} {chain:1s}"
        f"{resseq:4d}{icode:1s}   {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{occupancy:6.2f}{20.0:6.2f}          {element:>2s}  "
    )


def write_pdb(path, residues, *, models=None):
    def records(model_residues):
        lines = []
        serial = 1
        for residue in model_residues:
            resname, chain, resseq, icode, atoms = residue
            for atom in atoms:
                if len(atom) == 2:
                    atom_name, xyz = atom
                    altloc, occupancy = "", 1.0
                else:
                    atom_name, xyz, altloc, occupancy = atom
                lines.append(atom_line(serial, atom_name, resname, chain, resseq, xyz, icode=icode, altloc=altloc, occupancy=occupancy))
                serial += 1
        return lines

    if models is None:
        lines = records(residues)
    else:
        lines = []
        for index, model in enumerate(models, start=1):
            lines.append(f"MODEL     {index:4d}")
            lines.extend(records(model))
            lines.append("ENDMDL")
    path.write_text("\n".join(lines) + "\n")
    return path


def residue(resname, resseq, x, *, chain="A", icode="", missing=()):
    offsets = {"N": 0.0, "CA": 0.2, "C": 0.4, "O": 0.6}
    atoms = [(atom, (x + offset, 0.0, 0.0)) for atom, offset in offsets.items() if atom not in missing]
    return (resname, chain, resseq, icode, atoms)


def test_basic_rmsd_and_json_serializable(tmp_path):
    structure1 = write_pdb(tmp_path / "one.pdb", [residue("ALA", 1, 0), residue("GLY", 2, 2)])
    shifted = [residue("ALA", 1, 1), residue("GLY", 2, 3)]
    structure2 = write_pdb(tmp_path / "two.pdb", shifted)

    result = region_rmsd.analyze_regions(structure1, structure2, {"loop": "AG"})

    assert result["success"] is True
    assert result["regions"]["loop"]["ca"]["rmsd"] == pytest.approx(1.0)
    assert result["regions"]["loop"]["backbone"]["rmsd"] == pytest.approx(1.0)
    assert result["regions"]["loop"]["ca"]["atom_count"] == 2
    json.dumps(result)


def test_fixed_transform_is_applied_without_refitting(tmp_path):
    raw = [residue("ALA", 1, 0), residue("GLY", 2, 2)]
    structure1 = write_pdb(tmp_path / "raw.pdb", raw)
    rotation = np.diag([-1.0, -1.0, 1.0])
    translation = np.array([5.0, 2.0, 0.0])
    transformed = []
    for resname, chain, resseq, icode, atoms in raw:
        transformed.append((resname, chain, resseq, icode, [(name, rotation @ np.asarray(xyz) + translation) for name, xyz in atoms]))
    structure2 = write_pdb(tmp_path / "reference.pdb", transformed)

    result = region_rmsd.analyze_regions(
        structure1, structure2, {"loop": "AG"},
        rotation=rotation.tolist(), translation=translation.tolist(),
    )

    assert result["success"] is True
    assert result["combined"]["backbone"]["rmsd"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "resnames,query,code",
    [
        (["ALA", "GLY", "SER"], "GS", None),
        (["ALA", "GLY", "ALA", "GLY"], "AG", "region_not_unique"),
        (["ALA", "GLY", "SER"], "TT", "region_not_found"),
    ],
)
def test_unique_repeated_and_missing_regions(tmp_path, resnames, query, code):
    residues = [residue(name, index + 1, index * 2.0) for index, name in enumerate(resnames)]
    path1 = write_pdb(tmp_path / "one.pdb", residues)
    path2 = write_pdb(tmp_path / "two.pdb", residues)
    result = region_rmsd.analyze_regions(path1, path2, {"test": query})
    region = result["regions"]["test"]
    if code is None:
        assert region["success"] is True
    else:
        assert region["success"] is False
        assert region["error"]["code"] == code


def test_empty_or_different_sided_sequences_fail_structurally(tmp_path):
    residues = [residue("ALA", 1, 0)]
    path1 = write_pdb(tmp_path / "one.pdb", residues)
    path2 = write_pdb(tmp_path / "two.pdb", residues)
    empty = region_rmsd.analyze_regions(path1, path2, {"x": ""})
    mismatch = region_rmsd.analyze_regions(path1, path2, {"x": ("A", "G")})
    assert empty["regions"]["x"]["error"]["code"] == "empty_region"
    assert mismatch["regions"]["x"]["error"]["code"] == "region_sequence_mismatch"


def test_missing_backbone_fails_region_and_combined(tmp_path):
    complete = [residue("ALA", 1, 0)]
    incomplete = [residue("ALA", 1, 0, missing=("O",))]
    path1 = write_pdb(tmp_path / "one.pdb", incomplete)
    path2 = write_pdb(tmp_path / "two.pdb", complete)

    result = region_rmsd.analyze_regions(path1, path2, {"loop": "A"})

    region = result["regions"]["loop"]
    assert region["ca"]["success"] is True
    assert region["backbone"]["success"] is False
    assert region["backbone"]["error"]["missing"][0]["atom"] == "O"
    assert region["success"] is False
    assert result["combined"]["success"] is False


def test_insertion_code_negative_number_and_auditable_mapping(tmp_path):
    residues = [
        residue("MSE", -1, 0),
        residue("SEC", 10, 2, icode="A"),
        residue("PYL", 10, 4, icode="B"),
    ]
    path1 = write_pdb(tmp_path / "one.pdb", residues)
    path2 = write_pdb(tmp_path / "two.pdb", residues)

    result = region_rmsd.analyze_regions(path1, path2, {"special": "MUO"})

    mapping = result["regions"]["special"]["mapping"]
    assert [entry["structure1"]["resseq"] for entry in mapping] == [-1, 10, 10]
    assert [entry["structure1"]["icode"] for entry in mapping] == ["", "A", "B"]
    assert result["structure1"]["sequence"] == "MUO"


def test_only_first_model_is_parsed(tmp_path):
    first = [residue("ALA", 1, 0)]
    second = [residue("GLY", 1, 100)]
    path1 = write_pdb(tmp_path / "one.pdb", [], models=[first, second])
    path2 = write_pdb(tmp_path / "two.pdb", first)

    result = region_rmsd.analyze_regions(path1, path2, {"first": "A"})

    assert result["success"] is True
    assert result["structure1"]["sequence"] == "A"


def test_altloc_blank_then_a_then_highest_occupancy(tmp_path):
    atoms1 = [
        ("N", (0, 0, 0)), ("C", (0, 0, 0)), ("O", (0, 0, 0)),
        ("CA", (9, 0, 0), "B", 0.99),
        ("CA", (2, 0, 0), "A", 0.10),
        ("CA", (1, 0, 0), "", 0.01),
    ]
    atoms2 = [
        ("N", (0, 0, 0)), ("C", (0, 0, 0)), ("O", (0, 0, 0)),
        ("CA", (1, 0, 0)),
    ]
    one = [("ALA", "A", 1, "", atoms1)]
    two = [("ALA", "A", 1, "", atoms2)]
    path1 = write_pdb(tmp_path / "one.pdb", one)
    path2 = write_pdb(tmp_path / "two.pdb", two)
    result = region_rmsd.analyze_regions(path1, path2, {"x": "A"})
    assert result["regions"]["x"]["ca"]["rmsd"] == pytest.approx(0.0)

    a_vs_others = [("ALA", "A", 1, "", [
        ("N", (0, 0, 0)), ("C", (0, 0, 0)), ("O", (0, 0, 0)),
        ("CA", (3, 0, 0), "B", 0.99), ("CA", (2, 0, 0), "A", 0.01),
    ])]
    path3 = write_pdb(tmp_path / "three.pdb", a_vs_others)
    reference_a = [("ALA", "A", 1, "", [
        ("N", (0, 0, 0)), ("C", (0, 0, 0)), ("O", (0, 0, 0)), ("CA", (2, 0, 0)),
    ])]
    path4 = write_pdb(tmp_path / "four.pdb", reference_a)
    assert region_rmsd.analyze_regions(path3, path4, {"x": "A"})["regions"]["x"]["ca"]["rmsd"] == pytest.approx(0.0)

    occupancy_only = [("ALA", "A", 1, "", [
        ("N", (0, 0, 0)), ("C", (0, 0, 0)), ("O", (0, 0, 0)),
        ("CA", (4, 0, 0), "B", 0.25), ("CA", (5, 0, 0), "C", 0.75),
    ])]
    path5 = write_pdb(tmp_path / "five.pdb", occupancy_only)
    reference_c = [("ALA", "A", 1, "", [
        ("N", (0, 0, 0)), ("C", (0, 0, 0)), ("O", (0, 0, 0)), ("CA", (5, 0, 0)),
    ])]
    path6 = write_pdb(tmp_path / "six.pdb", reference_c)
    assert region_rmsd.analyze_regions(path5, path6, {"x": "A"})["regions"]["x"]["ca"]["rmsd"] == pytest.approx(0.0)


def test_unknown_residue_is_explicit_error(tmp_path):
    unknown = [("ZZZ", "A", 1, "", [("CA", (0, 0, 0))])]
    path1 = write_pdb(tmp_path / "one.pdb", unknown)
    path2 = write_pdb(tmp_path / "two.pdb", unknown)
    result = region_rmsd.analyze_regions(path1, path2, {"x": "A"})
    assert result["error"]["code"] == "unknown_residue"
    assert result["error"]["resname"] == "ZZZ"


def test_combined_uses_all_regions_atomwise(tmp_path):
    one = [residue("ALA", 1, 0), residue("GLY", 2, 3), residue("SER", 3, 6)]
    two = [residue("ALA", 1, 1), residue("GLY", 2, 3), residue("SER", 3, 9)]
    path1 = write_pdb(tmp_path / "one.pdb", one)
    path2 = write_pdb(tmp_path / "two.pdb", two)

    result = region_rmsd.analyze_regions(path1, path2, {"left": "A", "right": "S"})

    assert result["success"] is True
    assert result["combined"]["ca"]["atom_count"] == 2
    assert result["combined"]["ca"]["rmsd"] == pytest.approx(np.sqrt((1 + 9) / 2))
    assert result["combined"]["backbone"]["atom_count"] == 8


def test_cli_writes_json_and_returns_success(tmp_path):
    residues = [residue("ALA", 1, 0)]
    path1 = write_pdb(tmp_path / "one.pdb", residues)
    path2 = write_pdb(tmp_path / "two.pdb", residues)
    output = tmp_path / "result.json"

    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--structure1", str(path1), "--structure2", str(path2),
         "--region", "loop=A", "--out-json", str(output)],
        check=False, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["success"] is True
