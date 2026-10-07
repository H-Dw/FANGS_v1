import importlib.util
import json
import multiprocessing
import os
import socket
import time
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "tools" / "usalign_runner.py"
spec = importlib.util.spec_from_file_location("usalign_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(runner)


def pdb(path, xyz=(1.0, 2.0, 3.0)):
    path.write_text(
        "MODEL        1\n"
        f"ATOM      1  CA  ALA A   1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 20.00           C\n"
        "HETATM    2  O   HOH A   2       4.000   5.000   6.000  1.00 20.00           O\n"
        "ENDMDL\nMODEL        2\nATOM      3  CA  GLY B   1       9.000   9.000   9.000  1.00 20.00           C\nENDMDL\n"
    )


def fake_usalign(tmp_path, behavior="ok"):
    script = tmp_path / ("fake_" + behavior + ".py")
    body = """#!/usr/bin/env python3
import os, pathlib, sys, time
if len(sys.argv) == 2 and sys.argv[1] in ("-v", "--version", "-h"):
    print("USalign fake 20260813")
    raise SystemExit(0)
args=sys.argv[1:]
counter=os.environ.get("FAKE_USALIGN_COUNTER")
if counter:
    fd=os.open(counter, os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o644); os.write(fd,b"run\\n"); os.close(fd)
assert args[2:6] == ["-mol", "prot", "-ter", "2"]
assert "-m" in args and "-o" in args
prefix=pathlib.Path(args[args.index("-o")+1])
assert prefix.parent.name.startswith(".") and ".stage-" in prefix.parent.name
behavior=BEHAVIOR
if behavior == "timeout":
    time.sleep(2)
if behavior == "fail":
    print("intentional failure", file=sys.stderr); raise SystemExit(7)
matrix=pathlib.Path(args[args.index("-m")+1])
matrix.write_text("------ The rotation matrix to rotate Structure_1 to Structure_2 ------\\n"
                  " i          t(i)         u(i,1)         u(i,2)         u(i,3)\\n"
                  " 0      10.0000000000   0.0000000000  -1.0000000000   0.0000000000\\n"
                  " 1      20.0000000000   1.0000000000   0.0000000000   0.0000000000\\n"
                  " 2      30.0000000000   0.0000000000   0.0000000000   1.0000000000\\n")
(prefix.parent / (prefix.name + "_all_atm.pdb")).write_text("generated but deliberately unused\\n")
print("Aligned length= 42, RMSD= 1.25, Seq_ID=n_identical/n_aligned= 0.500")
print("TM-score= 0.75000 (if normalized by length of Structure_1)")
print("TM-score= 0.80000 (if normalized by length of Structure_2)")
""".replace("BEHAVIOR", repr(behavior))
    script.write_text(body)
    script.chmod(0o755)
    return script


def _mp_call(payload):
    args, kwargs = payload
    return runner.run_usalign(*args, **kwargs)


def common(tmp_path, behavior="ok"):
    a, b = tmp_path / "a.pdb", tmp_path / "b.pdb"
    pdb(a); pdb(b, (2, 3, 4))
    exe = fake_usalign(tmp_path, behavior)
    return a, b, exe, tmp_path / "cache"


def test_probe_version_extracts_real_banner(tmp_path):
    executable = tmp_path / "real_banner.py"
    executable.write_text("""#!/usr/bin/env python3
import sys
if sys.argv[1] == "-v":
    print("************************************************")
    print("US-align (Version 20240319)")
    print("Universal Structure Alignment by Chengxin Zhang")
    raise SystemExit(0)
raise SystemExit(2)
""")
    executable.chmod(0o755)
    version = runner._probe_version(executable, 2)
    assert version["text"] == "US-align (Version 20240319)"
    assert version["probe"]["flag"] == "-v"
    assert len(version["probe_attempts"]) == 1


def test_probe_version_ignores_failed_nonempty_output(tmp_path):
    executable = tmp_path / "failed_output.py"
    executable.write_text("""#!/usr/bin/env python3
import sys
if sys.argv[1] == "-v":
    print("Please provide structure B")
    raise SystemExit(1)
if sys.argv[1] == "--version":
    print("============================")
    print("US-align (Version 20250101)")
    raise SystemExit(0)
raise SystemExit(1)
""")
    executable.chmod(0o755)
    version = runner._probe_version(executable, 2)
    assert version["text"] == "US-align (Version 20250101)"
    assert version["probe"]["flag"] == "--version"
    assert version["probe_attempts"][0]["returncode"] == 1
    assert "Please provide structure B" in version["probe_attempts"][0]["output"]


def test_parse_matrix_and_metrics():
    metrics = runner.parse_usalign_stdout("Aligned length= 9, RMSD= 2.5, Seq_ID=n_identical/n_aligned= 0.25\nTM-score= 0.4\nTM-score= 0.6\n")
    assert metrics == {"aligned_length": 9, "global_rmsd": 2.5, "sequence_identity": .25, "tm_score1": .4, "tm_score2": .6}
    matrix = runner.parse_usalign_matrix("i t(i) u(i,1) u(i,2) u(i,3)\n0 1 1 0 0\n1 2 0 1 0\n2 3 0 0 1\n")
    assert matrix == {"translation": [1.0, 2.0, 3.0], "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}


def test_success_transform_and_cache_hit(tmp_path, monkeypatch):
    a, b, exe, cache = common(tmp_path)
    counter = tmp_path / "counter"
    monkeypatch.setenv("FAKE_USALIGN_COUNTER", str(counter))
    first = runner.run_usalign(a, b, usalign_path=exe, cache_dir=cache, pair_id="x")
    assert first["ok"] and not first["cache_hit"]
    assert first["metrics"]["global_rmsd"] == 1.25
    assert first["command"][3:7] == ["-mol", "prot", "-ter", "2"]
    aligned = Path(first["aligned_pdb"])
    lines = aligned.read_text().splitlines()
    atom = next(line for line in lines if line.startswith("ATOM"))
    assert tuple(float(atom[s:e]) for s, e in ((30, 38), (38, 46), (46, 54))) == (8.0, 21.0, 33.0)
    assert any(line.startswith("HETATM") for line in lines)
    second = runner.run_usalign(a, b, usalign_path=exe, cache_dir=cache, pair_id="x")
    assert second["ok"] and second["cache_hit"]
    assert counter.read_text().splitlines() == ["run"]


def test_concurrent_same_key_executes_once(tmp_path, monkeypatch):
    a, b, exe, cache = common(tmp_path)
    counter = tmp_path / "counter"
    monkeypatch.setenv("FAKE_USALIGN_COUNTER", str(counter))
    args = (str(a), str(b))
    kwargs = {"usalign_path": str(exe), "cache_dir": str(cache), "pair_id": "concurrent", "lock_timeout": 5}
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(2) as pool:
        results = pool.map(_mp_call, [(args, kwargs), (args, kwargs)])
    assert all(item["ok"] for item in results)
    assert sorted(item["cache_hit"] for item in results) == [False, True]
    assert counter.read_text().splitlines() == ["run"]


def test_stale_lock_reclaimed(tmp_path):
    a, b, exe, cache = common(tmp_path)
    cache.mkdir()
    version = runner._probe_version(exe.resolve(), 2)
    identity = {"pair_id": "stale", "structure1_id": str(a.resolve()), "structure2_id": str(b.resolve()),
                "structure1_sha256": runner._sha256_file(a), "structure2_sha256": runner._sha256_file(b),
                "usalign_path": str(exe.resolve()), "usalign_sha256": runner._sha256_file(exe),
                "usalign_version": version["text"], "parameters": runner.DEFAULT_PARAMETERS,
                "region_definition_fingerprint": runner.hashlib.sha256(runner._canonical_json(None).encode()).hexdigest(),
                "code_version": runner.CODE_VERSION, "schema_version": runner.SCHEMA_VERSION}
    key = runner.hashlib.sha256(runner._canonical_json(identity).encode()).hexdigest()
    (cache / (key + ".lock")).write_text(json.dumps({"pid": 999999999, "host": socket.gethostname(), "time": time.time()}))
    result = runner.run_usalign(a, b, usalign_path=exe, cache_dir=cache, pair_id="stale", lock_timeout=.5)
    assert result["ok"]


@pytest.mark.parametrize("behavior,error_type", [("fail", "process_error"), ("timeout", "timeout")])
def test_structured_failure_and_timeout(tmp_path, behavior, error_type):
    a, b, exe, cache = common(tmp_path, behavior)
    result = runner.run_usalign(a, b, usalign_path=exe, cache_dir=cache, timeout=.1)
    assert not result["ok"] and result["error"]["type"] == error_type
    assert result["schema_version"] == runner.SCHEMA_VERSION
    if behavior == "fail":
        assert result["returncode"] == 7 and "intentional failure" in result["stderr"]
