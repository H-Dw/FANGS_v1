#!/usr/bin/env python3
"""Concurrent-safe, content-addressed wrapper for USalign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
CODE_VERSION = "usalign-runner-2"
DEFAULT_PARAMETERS = ["-mol", "prot", "-ter", "2"]
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_id(path: Path, explicit: Optional[str]) -> str:
    return str(explicit) if explicit is not None else str(path.resolve())


def _version_text(output: str) -> Optional[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    version_pattern = re.compile(r"\bUS-?align\b.*\bVersion\b", re.IGNORECASE)
    for line in lines:
        if version_pattern.search(line):
            return line
    for line in lines:
        compact = re.sub(r"\s+", "", line)
        if compact and not re.fullmatch(r"[*#=_~+|:/\\.-]+", compact):
            return line
    return None


def _probe_version(executable: Path, timeout: float) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    for flag in ("-v", "--version", "-h"):
        try:
            proc = subprocess.run(
                [str(executable), flag], capture_output=True, text=True,
                timeout=max(0.1, min(timeout, 10.0)), check=False,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            attempt = {"flag": flag, "returncode": proc.returncode, "output": output}
            attempts.append(attempt)
            if proc.returncode == 0 and output.strip():
                version_text = _version_text(output)
                if version_text is not None:
                    return {"text": version_text, "probe": attempt, "probe_attempts": attempts}
        except (OSError, subprocess.TimeoutExpired) as exc:
            attempts.append({"flag": flag, "error": str(exc)})
    return {"text": "unknown", "probe_attempts": attempts}


def parse_usalign_stdout(text: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    summary = re.search(
        rf"Aligned\s+length\s*=\s*(\d+)\s*,\s*RMSD\s*=\s*({_FLOAT})\s*,\s*Seq_ID=n_identical/n_aligned\s*=\s*({_FLOAT})",
        text, re.IGNORECASE,
    )
    if summary:
        metrics.update(aligned_length=int(summary.group(1)), global_rmsd=float(summary.group(2)), sequence_identity=float(summary.group(3)))
    else:
        aligned = re.search(r"Aligned\s+length\s*=\s*(\d+)", text, re.IGNORECASE)
        rmsd = re.search(rf"\bRMSD\s*=\s*({_FLOAT})", text, re.IGNORECASE)
        identity = re.search(rf"Seq_ID(?:=n_identical/n_aligned)?\s*=\s*({_FLOAT})", text, re.IGNORECASE)
        if aligned: metrics["aligned_length"] = int(aligned.group(1))
        if rmsd: metrics["global_rmsd"] = float(rmsd.group(1))
        if identity: metrics["sequence_identity"] = float(identity.group(1))
    scores = [float(value) for value in re.findall(rf"TM-score\s*=\s*({_FLOAT})", text, re.IGNORECASE)]
    if scores:
        metrics["tm_score1"] = scores[0]
    if len(scores) > 1:
        metrics["tm_score2"] = scores[1]
    required = ("global_rmsd", "aligned_length", "tm_score1", "tm_score2", "sequence_identity")
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError("missing USalign metrics: " + ", ".join(missing))
    return metrics


def parse_usalign_matrix(text: str) -> Dict[str, List[List[float]]]:
    """Parse USalign's `i t(i) u(i,1) u(i,2) u(i,3)` matrix format.

    USalign defines Structure_1 -> Structure_2 as X' = t + U X.
    """
    rows: Dict[int, Tuple[float, List[float]]] = {}
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) < 5 or fields[0] not in {"0", "1", "2"}:
            continue
        try:
            idx = int(fields[0])
            values = [float(item) for item in fields[1:5]]
        except ValueError:
            continue
        rows[idx] = (values[0], values[1:])
    if set(rows) != {0, 1, 2}:
        raise ValueError("invalid USalign matrix; expected rows `i t(i) u(i,1) u(i,2) u(i,3)`")
    return {
        "translation": [rows[i][0] for i in range(3)],
        "rotation": [rows[i][1] for i in range(3)],
    }


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_text_fsync(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_fsync(path: Path, value: Any) -> None:
    _write_text_fsync(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _transform_pdb(source: Path, destination: Path, matrix: Dict[str, Any]) -> int:
    rotation = matrix["rotation"]
    translation = matrix["translation"]
    count = 0
    with source.open("r", encoding="utf-8", errors="replace") as inp, destination.open("w", encoding="utf-8", newline="") as out:
        for line in inp:
            if line.startswith(("ATOM  ", "HETATM")) and len(line.rstrip("\n\r")) >= 54:
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    values = [x, y, z]
                    moved = [translation[i] + sum(rotation[i][j] * values[j] for j in range(3)) for i in range(3)]
                    if any(abs(value) >= 10000 for value in moved):
                        raise ValueError("transformed coordinate exceeds PDB field width")
                    line = f"{line[:30]}{moved[0]:8.3f}{moved[1]:8.3f}{moved[2]:8.3f}{line[54:]}"
                    count += 1
                except ValueError as exc:
                    raise ValueError(f"invalid PDB coordinate line: {line.rstrip()}") from exc
            out.write(line)
        out.flush()
        os.fsync(out.fileno())
    if count == 0:
        raise ValueError("structure1 contains no parseable ATOM/HETATM coordinates")
    return count


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_is_stale(path: Path, ttl: float) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - float(data.get("time", path.stat().st_mtime))
        if data.get("host") == socket.gethostname() and not _pid_alive(int(data.get("pid", -1))):
            return True
        return ttl >= 0 and age > ttl
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        try:
            return ttl >= 0 and time.time() - path.stat().st_mtime > ttl
        except OSError:
            return False


def _acquire_lock(path: Path, timeout: float, ttl: float) -> str:
    deadline = time.monotonic() + timeout
    token = uuid.uuid4().hex
    payload = _canonical_json({"pid": os.getpid(), "host": socket.gethostname(), "time": time.time(), "token": token}) + "\n"
    while True:
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                os.write(fd, payload.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_dir(path.parent)
            return token
        except FileExistsError:
            if _lock_is_stale(path, ttl):
                tombstone = path.with_name(path.name + ".stale." + uuid.uuid4().hex)
                try:
                    os.replace(path, tombstone)
                    tombstone.unlink(missing_ok=True)
                    continue
                except FileNotFoundError:
                    continue
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for cache lock: {path}")
            time.sleep(min(0.05, max(0.005, deadline - time.monotonic())))


def _cache_result(cache_path: Path, key: str) -> Optional[Dict[str, Any]]:
    result_path = cache_path / "result.json"
    complete_path = cache_path / "COMPLETE"
    if not complete_path.is_file() or not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("schema_version") != SCHEMA_VERSION or result.get("cache_key") != key:
        return None
    for name in result.get("required_files", []):
        if not (cache_path / name).is_file():
            return None
    result["cache_hit"] = True
    return result


def _publish(stage: Path, cache_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "COMPLETE").unlink(missing_ok=True)
    _fsync_dir(cache_path)
    for child in stage.iterdir():
        if child.name in {"result.json", "COMPLETE"}:
            continue
        os.replace(child, cache_path / child.name)
    _fsync_dir(cache_path)
    _write_json_fsync(stage / "result.json", result)
    os.replace(stage / "result.json", cache_path / "result.json")
    _fsync_dir(cache_path)
    _write_text_fsync(stage / "COMPLETE", result["cache_key"] + "\n")
    os.replace(stage / "COMPLETE", cache_path / "COMPLETE")
    _fsync_dir(cache_path)
    return result


def run_usalign(
    structure1: os.PathLike[str] | str,
    structure2: os.PathLike[str] | str,
    *,
    usalign_path: os.PathLike[str] | str = "USalign",
    cache_dir: os.PathLike[str] | str = ".usalign_cache",
    pair_id: Optional[str] = None,
    structure1_id: Optional[str] = None,
    structure2_id: Optional[str] = None,
    timeout: float = 60.0,
    lock_timeout: float = 120.0,
    lock_ttl: float = 3600.0,
    region_definition: Any = None,
    extra_args: Sequence[str] = (),
) -> Dict[str, Any]:
    started = time.time()
    path1, path2 = Path(structure1), Path(structure2)
    executable = Path(usalign_path)
    cache_root = Path(cache_dir)
    try:
        path1, path2, executable = path1.resolve(strict=True), path2.resolve(strict=True), executable.resolve(strict=True)
        if not executable.is_file():
            raise FileNotFoundError(str(executable))
        cache_root.mkdir(parents=True, exist_ok=True)
        version = _probe_version(executable, timeout)
        parameters = DEFAULT_PARAMETERS + [str(item) for item in extra_args]
        identity = {
            "pair_id": pair_id,
            "structure1_id": _normalise_id(path1, structure1_id),
            "structure2_id": _normalise_id(path2, structure2_id),
            "structure1_sha256": _sha256_file(path1),
            "structure2_sha256": _sha256_file(path2),
            "usalign_path": str(executable),
            "usalign_sha256": _sha256_file(executable),
            "usalign_version": version["text"],
            "parameters": parameters,
            "region_definition_fingerprint": hashlib.sha256(_canonical_json(region_definition).encode()).hexdigest(),
            "code_version": CODE_VERSION,
            "schema_version": SCHEMA_VERSION,
        }
        key = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
        cache_path = cache_root / key
        cached = _cache_result(cache_path, key)
        if cached is not None:
            return cached
        lock_path = cache_root / (key + ".lock")
        lock_token = _acquire_lock(lock_path, lock_timeout, lock_ttl)
        try:
            cached = _cache_result(cache_path, key)
            if cached is not None:
                return cached
            stage = Path(tempfile.mkdtemp(prefix="." + key + ".stage-", dir=str(cache_root)))
            try:
                matrix_path = stage / "matrix.txt"
                output_prefix = stage / "usalign_output"
                command = [str(executable), str(path1), str(path2), *parameters, "-m", str(matrix_path), "-o", str(output_prefix)]
                try:
                    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
                    timed_out = False
                except subprocess.TimeoutExpired as exc:
                    proc = None
                    timed_out = True
                    stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                    stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                if not timed_out:
                    stdout, stderr = proc.stdout or "", proc.stderr or ""
                _write_text_fsync(stage / "stdout.txt", stdout)
                _write_text_fsync(stage / "stderr.txt", stderr)
                base: Dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION, "code_version": CODE_VERSION, "cache_key": key,
                    "cache_hit": False, "identity": identity, "command": command, "usalign_version": version,
                    "stdout": stdout, "stderr": stderr, "duration_seconds": time.time() - started,
                    "required_files": ["stdout.txt", "stderr.txt"],
                }
                if timed_out:
                    result = {**base, "ok": False, "error": {"type": "timeout", "message": f"USalign exceeded {timeout} seconds"}, "returncode": None}
                elif proc.returncode != 0:
                    result = {**base, "ok": False, "error": {"type": "process_error", "message": f"USalign exited with status {proc.returncode}"}, "returncode": proc.returncode}
                else:
                    metrics = parse_usalign_stdout(stdout)
                    matrix = parse_usalign_matrix(matrix_path.read_text(encoding="utf-8", errors="replace"))
                    _fsync_file(matrix_path)
                    atom_count = _transform_pdb(path1, stage / "aligned_structure1.pdb", matrix)
                    result = {**base, "ok": True, "returncode": proc.returncode, "metrics": metrics, "transform": matrix,
                              "aligned_pdb": str(cache_path / "aligned_structure1.pdb"), "transformed_atom_count": atom_count,
                              "required_files": ["stdout.txt", "stderr.txt", "matrix.txt", "aligned_structure1.pdb"]}
                return _publish(stage, cache_path, result)
            except Exception as exc:
                if 'base' not in locals():
                    base = {"schema_version": SCHEMA_VERSION, "code_version": CODE_VERSION, "cache_key": key, "cache_hit": False,
                            "identity": identity, "command": locals().get("command", []), "usalign_version": version,
                            "stdout": locals().get("stdout", ""), "stderr": locals().get("stderr", ""),
                            "duration_seconds": time.time() - started, "required_files": []}
                result = {**base, "ok": False, "returncode": getattr(locals().get("proc"), "returncode", None),
                          "error": {"type": "runner_error", "message": str(exc)}}
                return _publish(stage, cache_path, result)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        finally:
            try:
                lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
                if lock_data.get("token") == lock_token:
                    lock_path.unlink()
                    _fsync_dir(cache_root)
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                pass
    except Exception as exc:
        return {"schema_version": SCHEMA_VERSION, "code_version": CODE_VERSION, "ok": False, "cache_hit": False,
                "error": {"type": "setup_error", "message": str(exc)}, "duration_seconds": time.time() - started}


def _region_json(value: Optional[str]) -> Any:
    if value is None:
        return None
    candidate = Path(value)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    return json.loads(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure1")
    parser.add_argument("structure2")
    parser.add_argument("--usalign-path", default="USalign")
    parser.add_argument("--cache-dir", default=".usalign_cache")
    parser.add_argument("--pair-id")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--lock-timeout", type=float, default=120.0)
    parser.add_argument("--lock-ttl", type=float, default=3600.0)
    parser.add_argument("--region-definition-json")
    args = parser.parse_args(argv)
    try:
        region = _region_json(args.region_definition_json)
        result = run_usalign(args.structure1, args.structure2, usalign_path=args.usalign_path,
                             cache_dir=args.cache_dir, pair_id=args.pair_id, timeout=args.timeout,
                             lock_timeout=args.lock_timeout, lock_ttl=args.lock_ttl, region_definition=region)
    except Exception as exc:
        result = {"schema_version": SCHEMA_VERSION, "code_version": CODE_VERSION, "ok": False,
                  "cache_hit": False, "error": {"type": "cli_error", "message": str(exc)}}
    json.dump(result, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
