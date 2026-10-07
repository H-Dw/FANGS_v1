import csv
import importlib.util
import sys
from pathlib import Path

from Bio.PDB import PDBParser


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "tools" / "cluster_indi_mmseqs.py"
SPEC = importlib.util.spec_from_file_location("cluster_indi_mmseqs", MODULE_PATH)
cluster = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cluster
assert SPEC.loader is not None
SPEC.loader.exec_module(cluster)


def atom_line(serial, atom, resname, chain, resseq, x, *, icode=""):
    return (
        f"ATOM  {serial:5d} {atom:>4s} {resname:>3s} {chain:1s}"
        f"{resseq:4d}{icode:1s}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}  "
    )


def write_pdb(path, chains):
    lines = ["MODEL        1"]
    serial = 1
    for chain_id, residues in chains:
        for resname, resseq, icode in residues:
            for atom, offset in (("N", 0.0), ("CA", 0.2), ("C", 0.4), ("O", 0.6)):
                lines.append(atom_line(serial, atom, resname, chain_id, resseq, serial + offset, icode=icode))
                serial += 1
    lines.extend(("ENDMDL", "END"))
    path.write_text("\n".join(lines) + "\n")
    return path


def row(pdb_chain, sequence):
    return {
        "Sequence": sequence, "CDR1": "C1", "CDR2": "C2", "CDR3": "C3",
        "PDBChain": pdb_chain, "Member": pdb_chain,
    }


def test_default_sequence_source_is_tsv():
    args = cluster.parse_args([
        "--cdr-tsv", "cdr.tsv", "--pdb-dir", "pdbs", "--out-dir", "out",
    ])
    assert args.sequence_source == "tsv"
    assert args.fasta is None


def test_default_clustering_fasta_uses_tsv_sequence(tmp_path):
    rows = [row("oneA", " ac d "), row("twoB", "GG")]
    output = tmp_path / "clustering_sequences.fasta"

    sequences = cluster.build_clustering_fasta(rows, tmp_path, output)

    assert sequences == {"oneA": "ACD", "twoB": "GG"}
    assert output.read_text() == ">oneA\nACD\n>twoB\nGG\n"


def test_tsv_source_ignores_fasta_and_uses_sequence_column(tmp_path, capsys):
    external = tmp_path / "pdb2seq.fasta"
    external.write_text(">oneA\nQQQQ\n")
    output = tmp_path / "clustering_sequences.fasta"

    sequences = cluster.build_clustering_fasta(
        [row("oneA", "ACD")], tmp_path, output,
        existing_fasta=external, sequence_source="tsv",
    )

    captured = capsys.readouterr().out
    assert sequences == {"oneA": "ACD"}
    assert output.read_text() == ">oneA\nACD\n"
    assert "[WARNING] Ignoring --fasta" in captured
    assert "CDR TSV Sequence column" in captured


def test_fasta_sequence_source_uses_external_fasta(tmp_path):
    external = tmp_path / "external.fasta"
    external.write_text(">oneA\nq q\n")
    output = tmp_path / "clustering_sequences.fasta"

    sequences = cluster.build_clustering_fasta(
        [row("oneA", "TTT")], tmp_path, output,
        existing_fasta=external, sequence_source="fasta",
    )

    assert sequences == {"oneA": "QQ"}
    assert output.read_text() == ">oneA\nQQ\n"


def test_structure_source_can_use_fasta_cache(tmp_path):
    external = tmp_path / "external.fasta"
    external.write_text(">oneA\nq q\n")
    output = tmp_path / "clustering_sequences.fasta"

    sequences = cluster.build_clustering_fasta(
        [row("oneA", "TTT")], tmp_path, output,
        existing_fasta=external, sequence_source="structure",
    )

    assert sequences == {"oneA": "QQ"}
    assert output.read_text() == ">oneA\nQQ\n"


def test_structure_sequence_source_remains_available(tmp_path):
    write_pdb(tmp_path / "oneA.pdb", [("A", [("ALA", 1, ""), ("GLY", 2, "")])])
    output = tmp_path / "clustering_sequences.fasta"

    sequences = cluster.build_clustering_fasta(
        [row("oneA", "TTT")], tmp_path, output, sequence_source="structure"
    )

    assert sequences == {"oneA": "AG"}
    assert output.read_text() == ">oneA\nAG\n"


def test_extraction_selects_best_chain_and_internal_region_with_insertion_code(tmp_path):
    pdb_path = write_pdb(tmp_path / "rep.pdb", [
        ("B", [("ALA", 1, ""), ("CYS", 2, ""), ("GLY", 3, "")]),
        ("A", [
            ("THR", 8, ""), ("THR", 9, ""), ("ALA", 10, ""),
            ("CYS", 11, "A"), ("ASP", 12, ""), ("GLN", 13, ""),
        ]),
    ])
    output = tmp_path / "out.pdb"

    report = cluster.extract_representative_structure("rep", "ACD", pdb_path, output)

    assert report["structure_chain"] == "A"
    assert report["matched_residues"] == "3"
    assert report["missing_residues"] == "0"
    parsed = PDBParser(QUIET=True).get_structure("out", output)
    residues = list(next(parsed.get_models())["A"])
    assert [(res.get_resname(), res.id[1], res.id[2].strip()) for res in residues] == [
        ("ALA", 10, ""), ("CYS", 11, "A"), ("ASP", 12, ""),
    ]
    assert all(len(list(res.get_atoms())) == 4 for res in residues)
    assert output.read_text().rstrip().endswith("END")


def test_missing_target_residue_warns_and_is_reported_and_omitted(tmp_path, capsys):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    write_pdb(pdb_dir / "repA.pdb", [("X", [
        ("ALA", 1, ""), ("CYS", 2, ""), ("GLU", 4, ""),
    ])])
    output_dir = tmp_path / "structures"
    report_path = tmp_path / "cluster_rep_structure_extraction.tsv"

    rows = cluster.extract_representative_structures(
        ["repA"], {"repA": "ACDE"}, pdb_dir, output_dir, report_path
    )

    assert "[WARNING]" in capsys.readouterr().out
    assert rows[0]["missing_positions"] == "3:D"
    assert rows[0]["matched_residues"] == "3"
    assert rows[0]["missing_residues"] == "1"
    assert rows[0]["sequence_identity"] == "0.750000"
    with report_path.open(newline="") as handle:
        saved = list(csv.DictReader(handle, delimiter="\t"))
    assert saved == rows
    assert list(saved[0]) == list(cluster.EXTRACTION_COLUMNS)
    parsed = PDBParser(QUIET=True).get_structure("out", output_dir / "repA.pdb")
    assert [res.get_resname() for res in next(parsed.get_models())["X"]] == ["ALA", "CYS", "GLU"]


def test_rep_seq_fasta_uses_extracted_structure_not_original_pdb(tmp_path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    write_pdb(pdb_dir / "repA.pdb", [("A", [
        ("THR", 8, ""), ("THR", 9, ""), ("ALA", 10, ""),
        ("CYS", 11, "A"), ("ASP", 12, ""), ("GLN", 13, ""),
    ])])
    extraction_dir = tmp_path / "cluster_rep_structures"
    report_path = tmp_path / "cluster_rep_structure_extraction.tsv"
    fasta_path = tmp_path / "cluster_rep_seq.fasta"

    cluster.extract_representative_structures(
        ["repA"], {"repA": "ACD"}, pdb_dir, extraction_dir, report_path
    )
    sequences = cluster.write_rep_seq_fasta_from_extracted_structures(
        ["repA"], extraction_dir, fasta_path
    )

    original_sequence = cluster.get_sequence_from_pdb(str(pdb_dir / "repA.pdb"))
    assert original_sequence == "TTACDQ"
    assert sequences == {"repA": "ACD"}
    assert fasta_path.read_text() == ">repA\nACD\n"


def test_rep_seq_fasta_omits_residues_missing_from_extracted_structure(tmp_path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    write_pdb(pdb_dir / "repA.pdb", [("X", [
        ("ALA", 1, ""), ("CYS", 2, ""), ("GLU", 4, ""),
    ])])
    extraction_dir = tmp_path / "cluster_rep_structures"
    fasta_path = tmp_path / "cluster_rep_seq.fasta"

    cluster.extract_representative_structures(
        ["repA"], {"repA": "ACDE"}, pdb_dir, extraction_dir,
        tmp_path / "cluster_rep_structure_extraction.tsv",
    )
    sequences = cluster.write_rep_seq_fasta_from_extracted_structures(
        ["repA"], extraction_dir, fasta_path
    )

    assert sequences == {"repA": "ACE"}
    assert fasta_path.read_text() == ">repA\nACE\n"
