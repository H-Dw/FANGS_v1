import csv
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_pairwise_structures.py"
spec = importlib.util.spec_from_file_location("analyze_pairwise_structures", SCRIPT)
aps = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = aps
spec.loader.exec_module(aps)


def write_tsv(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_pdb(path, residues=("ALA", "GLY"), shift=0.0):
    lines = []
    for index, residue in enumerate(residues, 1):
        lines.append(
            f"ATOM  {index:5d}  CA  {residue:>3s} A{index:4d}    {index + shift:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C  \n"
        )
    path.write_text("".join(lines) + "END\n")


def successful_fake_worker(task):
    regions = {
        name: {
            "success": True,
            "mapping": [{"structure1": {"resseq": 1}, "structure2": {"resseq": 1}}],
            "ca": {"success": True, "rmsd": float(index + 1), "atom_count": 1},
            "backbone": {"success": True, "rmsd": float(index + 2), "atom_count": 4},
        }
        for index, name in enumerate(task["region_sequences"])
    }
    return {
        "schema_version": aps.SCHEMA_VERSION,
        "pair_id": task["pair_id"],
        "input_fingerprint": task["input_fingerprint"],
        "PDB1": task["member1"],
        "PDB2": task["member2"],
        "group_ids": task["group_ids"],
        "status": "success",
        "pair_status": "success",
        "usalign_status": "success",
        "usalign_global_rmsd": 1.25,
        "tm_score_1": 0.8,
        "tm_score_2": 0.7,
        "aligned_length": 100,
        "sequence_identity": 0.5,
        "ConnectorDistance": "",
        "combined_status": "success",
        "combined": {
            "success": True,
            "ca": {"success": True, "rmsd": 1.0, "atom_count": 2},
            "backbone": {"success": True, "rmsd": 2.0, "atom_count": 8},
        },
        "regions": regions,
    }


def sometimes_failing_worker(task):
    if task["member1"] == "A" and task["member2"] == "B":
        raise RuntimeError("synthetic failure")
    return successful_fake_worker(task)


def make_inputs(tmp_path, group_rows=None, ids=("A", "B", "C"), same_sequences=False):
    groups = tmp_path / "groups.tsv"
    metadata = tmp_path / "metadata.tsv"
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    group_rows = group_rows or [{"gid": "g1", "Member": "A,B,C"}]
    write_tsv(groups, ["gid", "Member"], group_rows)
    metadata_rows = []
    for index, identifier in enumerate(ids):
        metadata_rows.append({"ID": identifier, "CDR1": "AA", "CDR2": "GG"})
        residues = ("ALA", "GLY") if same_sequences else ("ALA", ("GLY", "SER", "VAL")[index % 3])
        write_pdb(pdb_dir / f"{identifier}.pdb", residues, shift=index)
    write_tsv(metadata, ["ID", "CDR1", "CDR2"], metadata_rows)
    return groups, metadata, pdb_dir


def read_output(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run_fake(tmp_path, *, workers=1, executor="thread", worker_fn=successful_fake_worker, resume=False, run_id="run", group_rows=None):
    groups, metadata, pdb_dir = make_inputs(tmp_path, group_rows=group_rows)
    out = tmp_path / "out.tsv"
    result = aps.run_analysis(
        groups=str(groups), metadata_tsv=str(metadata), pdb_dir=str(pdb_dir),
        usalign_path="/fake/USalign", out=str(out), group_id_col="gid",
        target_region_cols=["CDR1", "CDR2"], cache_dir=str(tmp_path / "cache"),
        run_id=run_id, workers=workers, executor=executor, max_in_flight=2,
        worker_fn=worker_fn, resume=resume,
    )
    return out, result


def test_pair_deduplication_and_group_ids(tmp_path):
    groups = tmp_path / "groups.tsv"
    write_tsv(groups, ["gid", "Member"], [
        {"gid": "z", "Member": "B,A,A,C"},
        {"gid": "a", "Member": "C,B"},
    ])
    parsed = aps.load_groups(groups, "Member", "gid")
    assert parsed[0]["members"] == ["B", "A", "C"]
    pairs = aps.build_pair_index(parsed)
    assert pairs == [
        {"member1": "A", "member2": "B", "group_ids": ["z"]},
        {"member1": "A", "member2": "C", "group_ids": ["z"]},
        {"member1": "B", "member2": "C", "group_ids": ["a", "z"]},
    ]


def test_generated_group_ids_are_stable(tmp_path):
    groups = tmp_path / "groups.tsv"
    write_tsv(groups, ["Member"], [{"Member": "B,A"}, {"Member": "C,A"}])
    first = aps.load_groups(groups)
    second = aps.load_groups(groups)
    assert [x["group_id"] for x in first] == [x["group_id"] for x in second]
    assert len({x["group_id"] for x in first}) == 2


def test_metadata_duplicate_rules_and_exact_regions(tmp_path):
    metadata = tmp_path / "metadata.tsv"
    rows = [{"ID": "A", "R1": "AAA", "R2": "BBB", "path": "a.pdb"}] * 2
    write_tsv(metadata, ["ID", "R1", "R2", "path"], rows)
    loaded = aps.load_metadata(metadata, "ID", ["R1", "R2"], "path")
    assert loaded["A"]["R2"] == "BBB"
    rows[1] = {**rows[1], "R2": "DIFFERENT"}
    write_tsv(metadata, ["ID", "R1", "R2", "path"], rows)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        aps.load_metadata(metadata, "ID", ["R1", "R2"], "path")


def test_embeddings_strict_validation(tmp_path):
    emb = tmp_path / "emb.tsv"
    write_tsv(emb, ["ID", "e0", "e1"], [{"ID": "A", "e0": "1", "e1": "2"}, {"ID": "B", "e0": "3", "e1": "4"}])
    assert aps.load_embeddings(emb)["B"] == [3.0, 4.0]
    write_tsv(emb, ["ID", "embedding"], [{"ID": "A", "embedding": "[1, nan]"}])
    with pytest.raises(ValueError, match="NaN/Inf"):
        aps.load_embeddings(emb)
    write_tsv(emb, ["ID", "embedding"], [{"ID": "A", "embedding": "1 2"}, {"ID": "A", "embedding": "1 3"}])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        aps.load_embeddings(emb)


@pytest.mark.parametrize("executor,workers", [("thread", 1), ("thread", 2), ("process", 2)])
def test_worker_count_and_executor_produce_identical_sorted_output(tmp_path, executor, workers):
    out, _ = run_fake(tmp_path, workers=workers, executor=executor)
    rows = read_output(out)
    assert [row["pair_id"] for row in rows] == sorted(row["pair_id"] for row in rows)
    assert len(rows) == 3
    assert len({row["pair_id"] for row in rows}) == 3
    assert all(row["CDR1_status"] == "success" for row in rows)


def test_bounded_executor_never_exceeds_limit():
    class Future:
        def __init__(self, value): self.value = value
        def result(self): return self.value
    class Executor:
        def __init__(self): self.outstanding = 0; self.maximum = 0
        def submit(self, fn, item):
            self.outstanding += 1
            self.maximum = max(self.maximum, self.outstanding)
            return Future(fn(item))
    executor = Executor()
    original_wait = aps.wait
    def fake_wait(pending, return_when=None):
        future = next(iter(pending))
        executor.outstanding -= 1
        return {future}, set(pending) - {future}
    aps.wait = fake_wait
    try:
        values = list(aps.bounded_executor_map(executor, lambda x: x * 2, range(9), 3))
    finally:
        aps.wait = original_wait
    assert executor.maximum == 3
    assert sorted(value for _, value, error in values if error is None) == list(range(0, 18, 2))


def test_checkpoint_resume_does_not_rerun_worker(tmp_path):
    out, first = run_fake(tmp_path)
    mtimes = {path.name: path.stat().st_mtime_ns for path in (Path(first["run_dir"]) / "pairs").glob("*.json")}
    def forbidden(_task):
        raise AssertionError("worker should not run")
    groups = tmp_path / "groups.tsv"
    metadata = tmp_path / "metadata.tsv"
    pdb_dir = tmp_path / "pdbs"
    aps.run_analysis(
        groups=str(groups), metadata_tsv=str(metadata), pdb_dir=str(pdb_dir), usalign_path="/fake/USalign",
        out=str(out), group_id_col="gid", target_region_cols=["CDR1", "CDR2"],
        cache_dir=str(tmp_path / "cache"), run_id="run", workers=2, executor="thread",
        worker_fn=forbidden, resume=True,
    )
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in (Path(first["run_dir"]) / "pairs").glob("*.json")}


def test_resume_rejects_fingerprint_mismatch(tmp_path):
    out, first = run_fake(tmp_path)
    checkpoint = next((Path(first["run_dir"]) / "pairs").glob("*.json"))
    value = json.loads(checkpoint.read_text())
    value["input_fingerprint"] = "bad"
    checkpoint.write_text(json.dumps(value))
    groups, metadata, pdb_dir = tmp_path / "groups.tsv", tmp_path / "metadata.tsv", tmp_path / "pdbs"
    with pytest.raises(ValueError, match="fingerprint"):
        aps.run_analysis(groups=str(groups), metadata_tsv=str(metadata), pdb_dir=str(pdb_dir), usalign_path="x", out=str(out),
                         group_id_col="gid", target_region_cols=["CDR1", "CDR2"], cache_dir=str(tmp_path / "cache"),
                         run_id="run", resume=True, executor="thread", worker_fn=successful_fake_worker)


def test_run_lock_active_and_safe_dead_pid_recovery(tmp_path):
    lock_path = tmp_path / "run.lock"
    active = {"pid": os.getpid(), "host": socket.gethostname(), "created_epoch": time.time()}
    lock_path.write_text(json.dumps(active))
    with pytest.raises(RuntimeError, match="active run lock"):
        aps.RunLock(lock_path).acquire()
    lock_path.write_text(json.dumps({**active, "pid": 99999999}))
    lock = aps.RunLock(lock_path)
    lock.acquire()
    assert lock.acquired
    lock.release()
    assert not lock_path.exists()


def test_failure_isolated_and_final_contains_all_pairs(tmp_path):
    out, result = run_fake(tmp_path, workers=2, worker_fn=sometimes_failing_worker)
    rows = read_output(out)
    assert len(rows) == 3
    assert sum(row["pair_status"] == "failed" for row in rows) == 1
    failed = next(row for row in rows if row["pair_status"] == "failed")
    assert failed["error_type"] == "RuntimeError"
    qc = json.loads(Path(result["qc_json"]).read_text())
    assert qc["status_counts"] == {"failed": 1, "success": 2}


def test_identical_full_atom_sequence_skips_without_optional_modules(tmp_path):
    groups, metadata_path, pdb_dir = make_inputs(tmp_path, ids=("A", "B"), group_rows=[{"gid": "g", "Member": "A,B"}], same_sequences=True)
    metadata = aps.load_metadata(metadata_path, "ID", ["CDR1"], None)
    tasks = aps.prepare_tasks(aps.build_pair_index(aps.load_groups(groups, "Member", "gid")), metadata, pdb_dir, ["CDR1"], None, None, 12, "missing", 1, tmp_path / "cache")
    result = aps.process_pair_task(tasks[0])
    assert result["status"] == "skipped_identical_sequence"
    aps.validate_checkpoint(result, tasks[0])


def test_region_sequence_mismatch_marks_pair_failed(monkeypatch, tmp_path):
    groups, metadata_path, pdb_dir = make_inputs(tmp_path, ids=("A", "B"), group_rows=[{"gid": "g", "Member": "A,B"}])
    write_tsv(metadata_path, ["ID", "CDR1"], [{"ID": "A", "CDR1": "AA"}, {"ID": "B", "CDR1": "BB"}])
    metadata = aps.load_metadata(metadata_path, "ID", ["CDR1"], None)
    task = aps.prepare_tasks(aps.build_pair_index(aps.load_groups(groups, "Member", "gid")), metadata, pdb_dir, ["CDR1"], None, None, 12, "x", 1, tmp_path / "cache")[0]
    monkeypatch.setattr(aps, "_run_usalign", lambda task: {"ok": True, "usalign_status": "success", "aligned_structure_path": task["structure_paths"][0]})
    monkeypatch.setattr(aps, "_run_regions", lambda task, global_result: {
        "success": False,
        "regions": {"CDR1": {"success": False, "error": {"code": "region_sequence_mismatch", "message": "Region sequences differ"}}},
        "combined": {"success": False},
    })
    result = aps.process_pair_task(task)
    assert result["status"] == "failed"
    assert result["regions"]["CDR1"]["error"]["code"] == "region_sequence_mismatch"


def test_cli_exposes_required_options():
    help_text = aps.build_parser().format_help()
    for option in ("--groups", "--metadata-tsv", "--target-region-cols", "--max-in-flight", "--plot-after-complete", "--export-aligned-dir"):
        assert option in help_text



def test_real_modules_import_from_final_version_root():
    code = """
import importlib.util
from pathlib import Path
path = Path('scripts/analyze_pairwise_structures.py')
spec = importlib.util.spec_from_file_location('aps_real_import', path)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module._import_optional('region_rmsd').__name__ == 'scripts.tools.region_rmsd'
assert module._import_optional('usalign_runner').__name__ == 'scripts.tools.usalign_runner'
assert module._import_optional('plot_pairwise_structure_results').__name__ == 'scripts.plot_pairwise_structure_results'
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=SCRIPT.parents[1], capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr


def test_structure_sequence_uses_region_parser_first_model_and_mse(tmp_path):
    first = tmp_path / "first.pdb"
    second = tmp_path / "second.pdb"
    first.write_text(
        "MODEL        1\n"
        "ATOM      1  CA  MSE A   1       1.000   0.000   0.000  1.00 20.00           C  \n"
        "ENDMDL\nMODEL        2\n"
        "ATOM      2  CA  GLY A   1       2.000   0.000   0.000  1.00 20.00           C  \nENDMDL\n"
    )
    second.write_text(
        "ATOM      1  CA  MET A   1       3.000   0.000   0.000  1.00 20.00           C  \n"
    )
    task = {
        "structure_paths": [str(first), str(second)],
        "schema_version": aps.SCHEMA_VERSION, "pair_id": "p", "input_fingerprint": "f",
        "member1": "A", "member2": "B", "group_ids": ["g"], "structure_sha256": ["1", "2"],
        "region_sequences": {}, "embeddings": None, "connector": 12, "usalign_path": "x", "timeout": 1, "cache_dir": str(tmp_path),
    }
    result = aps.process_pair_task(task)
    assert result["status"] == "skipped_identical_sequence"
    assert result["identical_structure_sequence"] is True


def test_structure_parse_failure_is_structured(tmp_path):
    bad = tmp_path / "bad.pdb"
    good = tmp_path / "good.pdb"
    bad.write_text("ATOM      1  CA  ZZZ A   1       1.000   0.000   0.000  1.00 20.00           C  \n")
    write_pdb(good, ("ALA",))
    task = {
        "structure_paths": [str(bad), str(good)], "pair_id": "p", "input_fingerprint": "f",
        "member1": "A", "member2": "B", "group_ids": ["g"], "structure_sha256": ["1", "2"],
        "region_sequences": {}, "embeddings": None, "connector": 12, "usalign_path": "x", "timeout": 1, "cache_dir": str(tmp_path),
    }
    result = aps.process_pair_task(task)
    assert result["pair_status"] == "failed"
    assert result["error_stage"] == "structure_parse"
    assert result["error_type"] == "unknown_residue"
    assert result["usalign_status"] == "not_run"


def test_future_failure_uses_main_schema():
    task = {"pair_id": "p", "input_fingerprint": "f", "member1": "A", "member2": "B", "group_ids": ["g"],
            "structure_paths": ["a", "b"], "structure_sha256": ["1", "2"]}
    result = aps._failure_from_future(task, RuntimeError("boom"))
    assert result["pair_status"] == "failed"
    assert result["error_stage"] == "executor"
    assert result["usalign_status"] == "not_run"
    assert result["combined_status"] == "failed"
    assert "GlobalStatus" not in result and "CombinedStatus" not in result


def test_combined_residue_count_and_region_error_aggregation():
    result = successful_fake_worker({"pair_id": "p", "input_fingerprint": "f", "member1": "A", "member2": "B",
                                     "group_ids": ["g"], "region_sequences": {"R1": ["A", "A"], "R2": ["G", "G"]}})
    row = aps._flatten_result(result, ["R1", "R2"])
    assert row["combined_n_residues"] == 2
    regional = {
        "success": False,
        "regions": {"R1": {"success": False, "error": {"code": "region_not_unique", "message": "not unique"}}},
        "combined": {"success": False},
    }
    assert aps._regional_failure(regional) == ("region_not_unique", "R1: not unique")


def test_export_aligned_versions_preserve_existing_content(tmp_path):
    source = tmp_path / "aligned.pdb"
    source.write_text("ATOM\n")
    destination = tmp_path / "exports"
    destination.mkdir()
    user_file = destination / "keep.txt"
    user_file.write_text("keep")
    aps._export_aligned([{"pair_id": "pair_one", "aligned_structure_path": str(source)}], destination)
    first_pointer = (destination / "CURRENT").read_text().strip()
    first_version = destination / first_pointer
    assert user_file.read_text() == "keep"
    assert (first_version / "pair_one.pdb").read_text() == "ATOM\n"
    time.sleep(0.002)
    aps._export_aligned([{"pair_id": "pair_two", "aligned_structure_path": str(source)}], destination)
    second_version = destination / (destination / "CURRENT").read_text().strip()
    assert second_version != first_version
    assert first_version.is_dir()
    assert user_file.read_text() == "keep"
    assert (second_version / "pair_two.pdb").is_file()


def test_default_workers_is_conservative():
    args = aps.build_parser().parse_args([
        "--groups", "g", "--metadata-tsv", "m", "--pdb-dir", "p", "--usalign-path", "u", "--out", "o",
    ])
    assert args.workers == min(4, os.cpu_count() or 1)



def test_embedding_duplicate_policies(tmp_path):
    emb = tmp_path / "duplicates.tsv"
    write_tsv(emb, ["ID", "e0", "e1"], [
        {"ID": "A", "e0": "1", "e1": "2"},
        {"ID": "A", "e0": "1", "e1": "3"},
        {"ID": "B", "e0": "4", "e1": "5"},
        {"ID": "B", "e0": "4", "e1": "5"},
    ])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        aps.load_embeddings(emb, duplicate_policy="error")
    assert aps.load_embeddings(emb, duplicate_policy="first")["A"] == [1.0, 2.0]
    assert aps.load_embeddings(emb, duplicate_policy="last")["A"] == [1.0, 3.0]
    assert aps.load_embeddings(emb, duplicate_policy="last")["B"] == [4.0, 5.0]
    with pytest.raises(ValueError, match="duplicate_policy"):
        aps.load_embeddings(emb, duplicate_policy="invalid")


def test_embedding_policy_changes_fingerprint_and_output_audit(tmp_path):
    groups, metadata_path, pdb_dir = make_inputs(
        tmp_path, ids=("A", "B"), group_rows=[{"gid": "g", "Member": "A,B"}],
    )
    metadata = aps.load_metadata(metadata_path, "ID", ["CDR1"], None)
    pairs = aps.build_pair_index(aps.load_groups(groups, "Member", "gid"))
    embeddings = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
    first = aps.prepare_tasks(pairs, metadata, pdb_dir, ["CDR1"], None, embeddings, 12, "x", 1, tmp_path / "cache", "first")[0]
    last = aps.prepare_tasks(pairs, metadata, pdb_dir, ["CDR1"], None, embeddings, 12, "x", 1, tmp_path / "cache", "last")[0]
    assert first["input_fingerprint"] != last["input_fingerprint"]
    assert first["embedding_duplicate_policy"] == "first"
    result = successful_fake_worker(first)
    result["embedding_duplicate_policy"] = first["embedding_duplicate_policy"]
    row = aps._flatten_result(result, ["CDR1"])
    assert row["embedding_duplicate_policy"] == "first"


def test_cli_embedding_duplicate_policy():
    parser = aps.build_parser()
    args = parser.parse_args([
        "--groups", "g", "--metadata-tsv", "m", "--pdb-dir", "p", "--usalign-path", "u", "--out", "o",
        "--embedding-duplicate-policy", "last",
    ])
    assert args.embedding_duplicate_policy == "last"
    assert parser.parse_args([
        "--groups", "g", "--metadata-tsv", "m", "--pdb-dir", "p", "--usalign-path", "u", "--out", "o",
    ]).embedding_duplicate_policy == "error"
