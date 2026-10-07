#!/usr/bin/env python3
"""Resumable, bounded pairwise structure-analysis coordinator.

The coordinator owns all shared output.  Workers receive one JSON-serialisable
pair task, call the optional ``tools.usalign_runner`` and ``tools.region_rmsd``
modules, and return one result dictionary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"success", "failed", "skipped_identical_sequence"}
BASE_COLUMNS = [
    "pair_id", "group_ids", "pdb1_id", "pdb2_id", "pdb1_path", "pdb2_path",
    "pair_status", "error_stage", "error_type", "error_message", "input_fingerprint",
    "embedding_duplicate_policy", "structure_sha256_1", "structure_sha256_2", "identical_structure_sequence",
    "usalign_status", "usalign_global_rmsd", "aligned_length", "tm_score_1", "tm_score_2",
    "sequence_identity", "usalign_cache_key", "usalign_cache_hit", "usalign_command",
    "usalign_version", "aligned_structure_path", "ConnectorDistance", "combined_status",
    "combined_n_residues", "combined_n_ca_atoms", "combined_n_backbone_atoms",
    "combined_ca_rmsd", "combined_backbone_rmsd",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def _read_tsv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        rows = [{key: _clean(value) for key, value in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = _clean(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def load_groups(path: str | Path, member_col: str = "Member", group_id_col: str | None = None) -> list[dict[str, Any]]:
    columns, rows = _read_tsv(path)
    if member_col not in columns:
        raise ValueError(f"groups TSV missing member column {member_col!r}")
    if group_id_col and group_id_col not in columns:
        raise ValueError(f"groups TSV missing group id column {group_id_col!r}")
    groups: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for zero_index, row in enumerate(rows):
        members = _dedupe_preserving_order(row[member_col].split(","))
        if len(members) < 2:
            continue
        if group_id_col:
            group_id = _clean(row[group_id_col])
            if not group_id:
                raise ValueError(f"empty group id at groups row {zero_index + 2}")
        else:
            representative = members[0]
            token = _sha256_bytes(f"{zero_index}\0{representative}".encode())[:10]
            group_id = f"g{zero_index + 1:06d}_{token}"
        if group_id in used_ids:
            raise ValueError(f"duplicate group id: {group_id}")
        used_ids.add(group_id)
        groups.append({"group_id": group_id, "members": members, "row_index": zero_index})
    return groups


def build_pair_index(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, str], set[str]] = {}
    for group in groups:
        for left, right in itertools.combinations(group["members"], 2):
            key = tuple(sorted((str(left), str(right))))
            pairs.setdefault(key, set()).add(str(group["group_id"]))
    return [
        {"member1": key[0], "member2": key[1], "group_ids": sorted(group_ids)}
        for key, group_ids in sorted(pairs.items())
    ]


def load_metadata(
    path: str | Path,
    id_col: str,
    region_cols: Sequence[str],
    structure_path_col: str | None = None,
) -> dict[str, dict[str, str]]:
    columns, rows = _read_tsv(path)
    required = [id_col, *region_cols] + ([structure_path_col] if structure_path_col else [])
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"metadata TSV missing columns: {', '.join(missing)}")
    result: dict[str, dict[str, str]] = {}
    for line_no, row in enumerate(rows, start=2):
        identifier = _clean(row[id_col])
        if not identifier:
            raise ValueError(f"empty metadata ID at row {line_no}")
        normalized = {column: _clean(row.get(column, "")) for column in columns}
        previous = result.get(identifier)
        if previous is not None and previous != normalized:
            raise ValueError(f"metadata ID {identifier!r} has conflicting duplicate rows")
        result[identifier] = normalized
    return result


def load_embeddings(
    path: str | Path, id_col: str = "ID", duplicate_policy: str = "error",
) -> dict[str, list[float]]:
    if duplicate_policy not in {"error", "first", "last"}:
        raise ValueError("embedding duplicate_policy must be error, first, or last")
    columns, rows = _read_tsv(path)
    if id_col not in columns:
        raise ValueError(f"embedding TSV missing ID column {id_col!r}")
    value_columns = [column for column in columns if column != id_col]
    if not value_columns:
        raise ValueError("embedding TSV has no embedding columns")
    result: dict[str, list[float]] = {}
    dimension: int | None = None
    for line_no, row in enumerate(rows, start=2):
        identifier = _clean(row[id_col])
        if not identifier:
            raise ValueError(f"empty embedding ID at row {line_no}")
        raw_values = [_clean(row[column]) for column in value_columns]
        if len(value_columns) == 1:
            text = raw_values[0].strip("[]() ")
            parts = [part for part in text.replace(",", " ").split() if part]
        else:
            parts = raw_values
        try:
            vector = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"non-numeric embedding for {identifier!r} at row {line_no}") from exc
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"embedding for {identifier!r} contains NaN/Inf or is empty")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError(f"embedding dimension mismatch for {identifier!r}: {len(vector)} != {dimension}")
        previous = result.get(identifier)
        if previous is not None:
            if previous == vector:
                continue
            if duplicate_policy == "error":
                raise ValueError(f"embedding ID {identifier!r} has conflicting duplicate rows")
            if duplicate_policy == "first":
                continue
        result[identifier] = vector
    return result


def _resolve_structure(member: str, metadata_row: Mapping[str, str], pdb_dir: Path, path_col: str | None) -> Path:
    raw = _clean(metadata_row.get(path_col, "")) if path_col else ""
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = pdb_dir / candidate
    else:
        candidate = pdb_dir / f"{member}.pdb"
    return candidate.resolve()


def _pair_id(member1: str, member2: str) -> str:
    digest = _sha256_bytes(f"{member1}\0{member2}".encode())[:20]
    return f"pair_{digest}"


def prepare_tasks(
    pair_index: Sequence[Mapping[str, Any]], metadata: Mapping[str, Mapping[str, str]],
    pdb_dir: str | Path, region_cols: Sequence[str], structure_path_col: str | None,
    embeddings: Mapping[str, Sequence[float]] | None, connector: float, usalign_path: str,
    timeout: float, cache_dir: str | Path, embedding_duplicate_policy: str = "error",
) -> list[dict[str, Any]]:
    pdb_dir = Path(pdb_dir)
    tasks: list[dict[str, Any]] = []
    for pair in pair_index:
        member1, member2 = str(pair["member1"]), str(pair["member2"])
        if member1 not in metadata or member2 not in metadata:
            missing = [member for member in (member1, member2) if member not in metadata]
            raise ValueError(f"members missing from metadata: {', '.join(missing)}")
        rows = (metadata[member1], metadata[member2])
        paths = (
            _resolve_structure(member1, rows[0], pdb_dir, structure_path_col),
            _resolve_structure(member2, rows[1], pdb_dir, structure_path_col),
        )
        missing_paths = [str(path) for path in paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"structure file(s) missing for {member1}/{member2}: {', '.join(missing_paths)}")
        shas = (sha256_file(paths[0]), sha256_file(paths[1]))
        region_sequences = {
            region: [_clean(rows[0].get(region)), _clean(rows[1].get(region))] for region in region_cols
        }
        emb_pair = None
        if embeddings is not None:
            if member1 not in embeddings or member2 not in embeddings:
                missing = [member for member in (member1, member2) if member not in embeddings]
                raise ValueError(f"members missing from embeddings: {', '.join(missing)}")
            emb_pair = [list(embeddings[member1]), list(embeddings[member2])]
        input_payload = {
            "members": [member1, member2], "structure_sha256": list(shas),
            "regions": region_sequences, "embeddings": emb_pair, "connector": connector,
            "embedding_duplicate_policy": embedding_duplicate_policy, "usalign_path": str(Path(usalign_path).expanduser()), "schema_version": SCHEMA_VERSION,
        }
        task = {
            "schema_version": SCHEMA_VERSION,
            "pair_id": _pair_id(member1, member2), "member1": member1, "member2": member2,
            "group_ids": list(pair["group_ids"]), "structure_paths": [str(paths[0]), str(paths[1])],
            "structure_sha256": list(shas), "region_sequences": region_sequences,
            "embeddings": emb_pair, "connector": connector,
            "embedding_duplicate_policy": embedding_duplicate_policy, "usalign_path": usalign_path,
            "timeout": timeout, "cache_dir": str(cache_dir),
            "input_fingerprint": _sha256_bytes(_stable_json(input_payload).encode()),
        }
        tasks.append(task)
    return sorted(tasks, key=lambda item: item["pair_id"])


def _import_optional(name: str):
    # Direct script execution puts ``scripts/`` on sys.path (``tools.*`` and
    # bare plotting module); package-style execution from final_version uses
    # ``scripts.tools.*`` and ``scripts.*``.
    candidates = [name, f"tools.{name}", f"scripts.{name}", f"scripts.tools.{name}", f"final_version.scripts.{name}", f"final_version.scripts.tools.{name}"]
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except ImportError as exc:
            parts = candidate.split(".")
            missing_candidates = {".".join(parts[:index]) for index in range(1, len(parts) + 1)}
            if exc.name not in missing_candidates:
                raise
    return None


def _parse_structure_sequences(task: Mapping[str, Any]) -> tuple[str, str]:
    module = _import_optional("region_rmsd")
    if module is None:
        raise RuntimeError("scripts.tools.region_rmsd is unavailable")
    parser = getattr(module, "parse_pdb", None)
    if not callable(parser):
        raise RuntimeError("region_rmsd.parse_pdb is unavailable")
    return tuple(parser(path).sequence for path in task["structure_paths"])

def _call_compatible(function: Callable[..., Any], kwargs: Mapping[str, Any], positional: Sequence[Any] = ()) -> Any:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*positional, **kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(*positional, **kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    required_positional = [p for p in signature.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
    if positional or len(required_positional) > len(accepted):
        return function(*positional, **accepted)
    return function(**accepted)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    if isinstance(value, (int, float)):
        return {"rmsd": float(value)}
    return {"value": value}


def _first(mapping: Mapping[str, Any], *names: str, default: Any = "") -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        if name in mapping:
            return mapping[name]
        if name.lower() in lowered:
            return lowered[name.lower()]
    return default


def _run_usalign(task: Mapping[str, Any]) -> dict[str, Any]:
    module = _import_optional("usalign_runner")
    if module is None:
        raise RuntimeError("scripts.tools.usalign_runner is unavailable")
    function = getattr(module, "run_usalign", None)
    if not callable(function):
        raise RuntimeError("usalign_runner.run_usalign is unavailable")
    p1, p2 = task["structure_paths"]
    data = function(
        p1, p2, usalign_path=task["usalign_path"], cache_dir=task["cache_dir"],
        pair_id=task["pair_id"], structure1_id=task["member1"], structure2_id=task["member2"],
        timeout=task["timeout"], region_definition=task["region_sequences"],
    )
    if not isinstance(data, Mapping):
        raise RuntimeError("usalign_runner returned a non-object result")
    data = dict(data)
    metrics = data.get("metrics") if isinstance(data.get("metrics"), Mapping) else {}
    error = data.get("error") if isinstance(data.get("error"), Mapping) else {}
    return {
        "ok": bool(data.get("ok")), "usalign_status": "success" if data.get("ok") else "failed",
        "usalign_global_rmsd": metrics.get("global_rmsd", ""),
        "aligned_length": metrics.get("aligned_length", ""), "tm_score_1": metrics.get("tm_score1", ""),
        "tm_score_2": metrics.get("tm_score2", ""), "sequence_identity": metrics.get("sequence_identity", ""),
        "aligned_structure_path": data.get("aligned_pdb", ""), "transform": data.get("transform"),
        "usalign_cache_key": data.get("cache_key", ""), "usalign_cache_hit": data.get("cache_hit", ""),
        "usalign_command": data.get("command", []), "usalign_version": data.get("usalign_version", {}),
        "error_type": error.get("type", ""), "error_message": error.get("message", ""), "usalign_raw": data,
    }


def _run_regions(task: Mapping[str, Any], global_result: Mapping[str, Any]) -> dict[str, Any]:
    module = _import_optional("region_rmsd")
    if module is None:
        raise RuntimeError("scripts.tools.region_rmsd is unavailable")
    function = getattr(module, "analyze_regions", None)
    if not callable(function):
        raise RuntimeError("region_rmsd.analyze_regions is unavailable")
    region_specs = {
        name: {"structure1": sequences[0], "structure2": sequences[1]}
        for name, sequences in task["region_sequences"].items()
    }
    aligned = global_result.get("aligned_structure_path")
    if not aligned:
        raise RuntimeError("USalign result did not provide aligned_structure_path")
    data = function(aligned, task["structure_paths"][1], region_specs)
    if not isinstance(data, Mapping):
        raise RuntimeError("region_rmsd returned a non-object result")
    return dict(data)

def _regional_failure(regional: Mapping[str, Any]) -> tuple[str, str]:
    top_error = regional.get("error")
    if isinstance(top_error, Mapping):
        return _clean(top_error.get("code") or "region_error"), _clean(top_error.get("message") or _stable_json(top_error))
    for region_name, value in regional.get("regions", {}).items():
        if not isinstance(value, Mapping) or value.get("success"):
            continue
        candidates = [value.get("error")]
        for metric_name in ("ca", "backbone"):
            metric = value.get(metric_name)
            if isinstance(metric, Mapping) and not metric.get("success"):
                candidates.append(metric.get("error"))
        for error in candidates:
            if isinstance(error, Mapping):
                code = _clean(error.get("code") or "region_error")
                message = _clean(error.get("message") or _stable_json(error))
                return code, f"{region_name}: {message}"
        return "region_failed", f"{region_name}: region analysis failed"
    combined = regional.get("combined")
    if isinstance(combined, Mapping) and not combined.get("success"):
        error = combined.get("error")
        if isinstance(error, Mapping):
            return _clean(error.get("code") or "combined_error"), _clean(error.get("message") or _stable_json(error))
        return "combined_failed", "combined region analysis failed"
    return "region_rmsd_failed", "region RMSD analysis failed without a detailed error"


def process_pair_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Top-level pickleable worker; never writes run-level shared files."""
    base = {
        "schema_version": SCHEMA_VERSION, "pair_id": task["pair_id"],
        "input_fingerprint": task["input_fingerprint"], "PDB1": task["member1"], "PDB2": task["member2"],
        "group_ids": list(task["group_ids"]), "structure_path_1": task["structure_paths"][0],
        "structure_path_2": task["structure_paths"][1],
        "embedding_duplicate_policy": task.get("embedding_duplicate_policy", "error"),
        "structure_sha256_1": task["structure_sha256"][0],
        "structure_sha256_2": task["structure_sha256"][1], "started_at": _utc_now(),
    }
    try:
        try:
            seq1, seq2 = _parse_structure_sequences(task)
        except Exception as exc:
            error_code = _clean(getattr(exc, "code", "")) or type(exc).__name__
            error_message = _clean(getattr(exc, "message", "")) or str(exc)
            return {**base, "status": "failed", "pair_status": "failed", "error_stage": "structure_parse",
                    "error_type": error_code, "error_message": error_message, "usalign_status": "not_run",
                    "combined_status": "failed", "regions": {}, "finished_at": _utc_now()}
        if seq1 and seq1 == seq2:
            return {**base, "status": "skipped_identical_sequence", "pair_status": "skipped_identical_structure_sequence",
                    "identical_structure_sequence": True, "usalign_status": "skipped", "combined_status": "skipped",
                    "regions": {}, "finished_at": _utc_now()}
        connector_distance = ""
        if task.get("embeddings") is not None:
            left, right = task["embeddings"]
            connector_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))) / math.sqrt(float(task["connector"]))
        global_result = _run_usalign(task)
        if not global_result["ok"]:
            return {**base, **global_result, "status": "failed", "pair_status": "failed", "error_stage": "usalign",
                    "ConnectorDistance": connector_distance, "combined_status": "failed", "regions": {}, "finished_at": _utc_now()}
        regional = _run_regions(task, global_result)
        regions = dict(regional.get("regions", {}))
        combined = regional.get("combined") if isinstance(regional.get("combined"), Mapping) else {}
        all_ok = bool(regional.get("success"))
        error_type, error_message = ("", "") if all_ok else _regional_failure(regional)
        return {**base, **global_result, "status": "success" if all_ok else "failed",
                "pair_status": "success" if all_ok else "failed", "error_stage": "" if all_ok else "region_rmsd",
                "error_type": error_type, "error_message": error_message,
                "identical_structure_sequence": False, "ConnectorDistance": connector_distance, "regions": regions,
                "combined": dict(combined), "combined_status": "success" if combined.get("success") else "failed",
                "finished_at": _utc_now()}
    except Exception as exc:
        return {**base, "status": "failed", "pair_status": "failed", "error_stage": "worker",
                "error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc(),
                "usalign_status": "failed", "combined_status": "failed", "regions": {}, "finished_at": _utc_now()}

def validate_checkpoint(value: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("checkpoint is not an object")
    required = {"schema_version", "pair_id", "input_fingerprint", "status", "PDB1", "PDB2", "group_ids", "regions"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"checkpoint missing keys: {sorted(missing)}")
    if value["schema_version"] != SCHEMA_VERSION or value["pair_id"] != task["pair_id"]:
        raise ValueError("checkpoint schema/pair key mismatch")
    if value["input_fingerprint"] != task["input_fingerprint"]:
        raise ValueError("checkpoint input fingerprint mismatch")
    if value["PDB1"] != task["member1"] or value["PDB2"] != task["member2"]:
        raise ValueError("checkpoint member key mismatch")
    if sorted(value["group_ids"]) != sorted(task["group_ids"]):
        raise ValueError("checkpoint group_ids mismatch")
    if value["status"] not in TERMINAL_STATUSES:
        raise ValueError(f"checkpoint is not terminal: {value['status']}")
    if not isinstance(value["regions"], dict):
        raise ValueError("checkpoint regions is not an object")
    return value


@dataclass
class RunLock:
    path: Path
    ttl_seconds: float | None = None
    acquired: bool = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "host": socket.gethostname(), "created_at": _utc_now(), "created_epoch": time.time()}
        while True:
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text())
                except Exception as exc:
                    raise RuntimeError(f"run lock exists and is unreadable: {self.path}") from exc
                same_host = existing.get("host") == socket.gethostname()
                dead = False
                if same_host and isinstance(existing.get("pid"), int):
                    try:
                        os.kill(existing["pid"], 0)
                    except ProcessLookupError:
                        dead = True
                    except PermissionError:
                        dead = False
                expired = self.ttl_seconds is not None and time.time() - float(existing.get("created_epoch", time.time())) > self.ttl_seconds
                if (same_host and dead) or expired:
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                raise RuntimeError(f"active run lock exists: {self.path} (pid={existing.get('pid')}, host={existing.get('host')})")
            else:
                with os.fdopen(fd, "w") as handle:
                    json.dump(payload, handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                self.acquired = True
                return

    def release(self) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def bounded_executor_map(executor: Any, function: Callable[[Any], Any], items: Iterable[Any], max_in_flight: int) -> Iterator[tuple[Any, Any, BaseException | None]]:
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    iterator = iter(items)
    pending: dict[Any, Any] = {}
    exhausted = False
    while pending or not exhausted:
        while not exhausted and len(pending) < max_in_flight:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending[executor.submit(function, item)] = item
        if not pending:
            break
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            item = pending.pop(future)
            try:
                yield item, future.result(), None
            except BaseException as exc:
                yield item, None, exc


def _failure_from_future(task: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "pair_id": task["pair_id"], "input_fingerprint": task["input_fingerprint"],
        "PDB1": task["member1"], "PDB2": task["member2"], "group_ids": task["group_ids"], "status": "failed",
        "pair_status": "failed", "error_stage": "executor", "error_type": type(exc).__name__,
        "error_message": str(exc), "usalign_status": "not_run", "combined_status": "failed",
        "regions": {}, "finished_at": _utc_now(), "structure_path_1": task["structure_paths"][0],
        "structure_path_2": task["structure_paths"][1],
        "embedding_duplicate_policy": task.get("embedding_duplicate_policy", "error"),
        "structure_sha256_1": task["structure_sha256"][0],
        "structure_sha256_2": task["structure_sha256"][1],
    }


def _manifest(tasks: Sequence[Mapping[str, Any]], completed: int, run_id: str) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id, "expected_pairs": len(tasks), "completed_pairs": completed,
            "pair_ids": [task["pair_id"] for task in tasks], "updated_at": _utc_now()}


def _error_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return _clean(value.get("message") or value.get("code") or _stable_json(value))
    return _clean(value)


def _flatten_result(result: Mapping[str, Any], region_cols: Sequence[str]) -> dict[str, Any]:
    row = {
        "pair_id": result.get("pair_id", ""), "group_ids": ",".join(sorted(result.get("group_ids", []))),
        "pdb1_id": result.get("PDB1", ""), "pdb2_id": result.get("PDB2", ""),
        "pdb1_path": result.get("structure_path_1", ""), "pdb2_path": result.get("structure_path_2", ""),
        "pair_status": result.get("pair_status", result.get("status", "")), "error_stage": result.get("error_stage", ""),
        "error_type": result.get("error_type", ""), "error_message": result.get("error_message", ""),
        "input_fingerprint": result.get("input_fingerprint", ""),
        "embedding_duplicate_policy": result.get("embedding_duplicate_policy", "error"),
        "structure_sha256_1": result.get("structure_sha256_1", ""), "structure_sha256_2": result.get("structure_sha256_2", ""),
        "identical_structure_sequence": result.get("identical_structure_sequence", ""),
        "usalign_status": result.get("usalign_status", ""), "usalign_global_rmsd": result.get("usalign_global_rmsd", ""),
        "aligned_length": result.get("aligned_length", ""), "tm_score_1": result.get("tm_score_1", ""),
        "tm_score_2": result.get("tm_score_2", ""), "sequence_identity": result.get("sequence_identity", ""),
        "usalign_cache_key": result.get("usalign_cache_key", ""), "usalign_cache_hit": result.get("usalign_cache_hit", ""),
        "usalign_command": _stable_json(result.get("usalign_command", [])),
        "usalign_version": _stable_json(result.get("usalign_version", {})),
        "aligned_structure_path": result.get("aligned_structure_path", ""), "ConnectorDistance": result.get("ConnectorDistance", ""),
    }
    combined = result.get("combined") if isinstance(result.get("combined"), Mapping) else {}
    combined_ca = combined.get("ca") if isinstance(combined.get("ca"), Mapping) else {}
    combined_bb = combined.get("backbone") if isinstance(combined.get("backbone"), Mapping) else {}
    row.update({
        "combined_status": result.get("combined_status", "success" if combined.get("success") else "failed"),
        "combined_n_residues": combined_ca.get("atom_count", "") or sum(
            len(value.get("mapping", [])) for value in result.get("regions", {}).values()
            if isinstance(value, Mapping) and value.get("success") and isinstance(value.get("mapping"), list)
        ),
        "combined_n_ca_atoms": combined_ca.get("atom_count", ""), "combined_n_backbone_atoms": combined_bb.get("atom_count", ""),
        "combined_ca_rmsd": combined_ca.get("rmsd", ""), "combined_backbone_rmsd": combined_bb.get("rmsd", ""),
    })
    for region in region_cols:
        value = result.get("regions", {}).get(region, {})
        ca = value.get("ca") if isinstance(value.get("ca"), Mapping) else {}
        bb = value.get("backbone") if isinstance(value.get("backbone"), Mapping) else {}
        mapping = value.get("mapping") if isinstance(value.get("mapping"), list) else []
        row.update({
            f"{region}_status": "success" if value.get("success") else "failed",
            f"{region}_residue_ids_1": _stable_json([item.get("structure1", {}) for item in mapping]),
            f"{region}_residue_ids_2": _stable_json([item.get("structure2", {}) for item in mapping]),
            f"{region}_n_residues": len(mapping) if mapping else "", f"{region}_n_ca_atoms": ca.get("atom_count", ""),
            f"{region}_n_backbone_atoms": bb.get("atom_count", ""), f"{region}_ca_rmsd": ca.get("rmsd", ""),
            f"{region}_backbone_rmsd": bb.get("rmsd", ""), f"{region}_error": _error_text(value.get("error") or ca.get("error") or bb.get("error")),
        })
    return row

def _publish_tsv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try: os.unlink(tmp_name)
        except FileNotFoundError: pass
        raise


def _publish_qc(out_path: Path, results: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    counts = Counter(str(result.get("status", "unknown")) for result in results)
    summary = {"schema_version": SCHEMA_VERSION, "total": len(results), "status_counts": dict(sorted(counts.items())), "generated_at": _utc_now()}
    json_path = out_path.with_suffix(out_path.suffix + ".qc.json")
    tsv_path = out_path.with_suffix(out_path.suffix + ".qc.tsv")
    atomic_write_json(json_path, summary)
    _publish_tsv(tsv_path, [{"status": key, "count": value} for key, value in sorted(counts.items())], ["status", "count"])
    return json_path, tsv_path


def _copy_aligned_files(results: Sequence[Mapping[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for result in results:
        source_text = _clean(result.get("aligned_structure_path"))
        if not source_text:
            continue
        source = Path(source_text)
        if source.is_file():
            suffix = "".join(source.suffixes) or ".pdb"
            shutil.copy2(source, directory / f"{result['pair_id']}{suffix}")


def _export_aligned(results: Sequence[Mapping[str, Any]], destination: Path) -> None:
    """Publish aligned structures without replacing or deleting user content."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
    if not destination.exists():
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.export-", dir=str(destination.parent)))
        try:
            versions = staging / ".versions"
            version = versions / version_id
            versions.mkdir()
            _copy_aligned_files(results, version)
            _atomic_bytes(staging / "CURRENT", f".versions/{version_id}\n".encode())
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return
    if not destination.is_dir():
        raise ValueError(f"aligned export destination is not a directory: {destination}")
    versions = destination / ".versions"
    versions.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{version_id}.staging-", dir=str(versions)))
    try:
        # tempfile created the directory; populate it directly.
        for result in results:
            source_text = _clean(result.get("aligned_structure_path"))
            if not source_text:
                continue
            source = Path(source_text)
            if source.is_file():
                suffix = "".join(source.suffixes) or ".pdb"
                shutil.copy2(source, staging / f"{result['pair_id']}{suffix}")
        published = versions / version_id
        os.replace(staging, published)
        _atomic_bytes(destination / "CURRENT", f".versions/{version_id}\n".encode())
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

def run_analysis(
    *, groups: str, metadata_tsv: str, pdb_dir: str, usalign_path: str, out: str,
    member_col: str = "Member", group_id_col: str | None = None, metadata_id_col: str = "ID",
    structure_path_col: str | None = None, target_region_cols: Sequence[str] = (), emb: str | None = None,
    embedding_id_col: str = "ID", embedding_duplicate_policy: str = "error",
    connector: float = 12.0, cache_dir: str | None = None,
    resume: bool = False, run_id: str | None = None, export_aligned_dir: str | None = None,
    timeout: float = 60.0, workers: int = 1, executor: str = "process", max_in_flight: int | None = None,
    lock_ttl: float | None = None, worker_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = process_pair_task,
    plot_after_complete: bool = False, plot_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if connector <= 0 or workers < 1:
        raise ValueError("connector and workers must be positive")
    max_in_flight = max_in_flight or max(1, workers * 2)
    output = Path(out).resolve()
    cache_root = Path(cache_dir).resolve() if cache_dir else output.parent / ".pairwise_cache"
    if run_id is None:
        run_id = _sha256_bytes(_stable_json({"groups": str(Path(groups).resolve()), "metadata": str(Path(metadata_tsv).resolve()), "out": str(output)}).encode())[:16]
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and hyphen")
    run_dir = cache_root / "runs" / run_id
    checkpoints = run_dir / "pairs"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = RunLock(run_dir / "run.lock", ttl_seconds=lock_ttl)
    with lock:
        group_rows = load_groups(groups, member_col, group_id_col)
        pair_index = build_pair_index(group_rows)
        metadata = load_metadata(metadata_tsv, metadata_id_col, target_region_cols, structure_path_col)
        embeddings = load_embeddings(emb, embedding_id_col, embedding_duplicate_policy) if emb else None
        tasks = prepare_tasks(
            pair_index, metadata, pdb_dir, target_region_cols, structure_path_col, embeddings,
            connector, usalign_path, timeout, cache_root / "content", embedding_duplicate_policy,
        )
        checkpoints.mkdir(parents=True, exist_ok=True)
        pending: list[dict[str, Any]] = []
        completed = 0
        for task in tasks:
            checkpoint = checkpoints / f"{task['pair_id']}.json"
            if resume and checkpoint.is_file():
                try:
                    validate_checkpoint(json.loads(checkpoint.read_text()), task)
                except Exception as exc:
                    raise ValueError(f"invalid resume checkpoint {checkpoint}: {exc}") from exc
                completed += 1
            else:
                pending.append(task)
        atomic_write_json(run_dir / "manifest.json", _manifest(tasks, completed, run_id))
        executor_class = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
        if executor not in {"process", "thread"}:
            raise ValueError("executor must be process or thread")
        with executor_class(max_workers=workers) as pool:
            for task, result, error in bounded_executor_map(pool, worker_fn, pending, max_in_flight):
                value = _failure_from_future(task, error) if error is not None else dict(result)
                try:
                    validate_checkpoint(value, task)
                except Exception as exc:
                    value = _failure_from_future(task, RuntimeError(f"worker returned invalid result: {exc}"))
                atomic_write_json(checkpoints / f"{task['pair_id']}.json", value)
                completed += 1
                atomic_write_json(run_dir / "manifest.json", _manifest(tasks, completed, run_id))
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for task in tasks:
            checkpoint = checkpoints / f"{task['pair_id']}.json"
            if not checkpoint.is_file():
                raise RuntimeError(f"missing terminal checkpoint: {checkpoint}")
            value = validate_checkpoint(json.loads(checkpoint.read_text()), task)
            if value["pair_id"] in seen:
                raise RuntimeError(f"duplicate final pair_id: {value['pair_id']}")
            seen.add(value["pair_id"])
            results.append(value)
        results.sort(key=lambda item: item["pair_id"])
        dynamic = [f"{region}_{suffix}" for region in target_region_cols for suffix in (
            "status", "residue_ids_1", "residue_ids_2", "n_residues", "n_ca_atoms", "n_backbone_atoms",
            "ca_rmsd", "backbone_rmsd", "error",
        )]
        _publish_tsv(output, [_flatten_result(value, target_region_cols) for value in results], [*BASE_COLUMNS, *dynamic])
        qc_json, qc_tsv = _publish_qc(output, results)
        if export_aligned_dir:
            _export_aligned(results, Path(export_aligned_dir).resolve())
    if plot_after_complete:
        if not (plot_kwargs or {}).get("output_dir"):
            raise ValueError("--plot-after-complete requires --plot-out-dir")
        module = _import_optional("plot_pairwise_structure_results")
        if module is None:
            raise RuntimeError("plot_pairwise_structure_results is unavailable")
        function = next((getattr(module, name) for name in ("plot_results", "plot_pairwise_results", "main") if callable(getattr(module, name, None))), None)
        if function is None:
            raise RuntimeError("plot module exposes no supported entry point")
        kwargs = {"input_path": str(output), **dict(plot_kwargs or {})}
        _call_compatible(function, kwargs)
    return {"out": str(output), "qc_json": str(qc_json), "qc_tsv": str(qc_tsv), "run_dir": str(run_dir), "pair_count": len(results)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--metadata-tsv", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--usalign-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--member-col", default="Member")
    parser.add_argument("--group-id-col")
    parser.add_argument("--metadata-id-col", default="ID")
    parser.add_argument("--structure-path-col")
    parser.add_argument("--target-region-cols", nargs="+", default=[])
    parser.add_argument("--emb")
    parser.add_argument("--embedding-id-col", default="ID")
    parser.add_argument("--embedding-duplicate-policy", choices=("error", "first", "last"), default="error")
    parser.add_argument("--connector", type=float, default=12.0)
    parser.add_argument(
        "--alignment-mode",
        choices=("whole-chain-usalign", "fr-ca-kabsch"),
        default="whole-chain-usalign",
        help="Primary alignment used for regional RMSD analysis",
    )
    parser.add_argument("--cdr-definition", default="IMGT")
    parser.add_argument("--qc-min-coverage", type=float, default=0.80)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--permutation-resamples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260831)
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--export-aligned-dir")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--lock-ttl", type=float, help="Seconds after which a lock may be reclaimed")
    parser.add_argument("--plot-after-complete", action="store_true")
    parser.add_argument("--plot-out-dir")
    parser.add_argument("--plot-metrics", nargs="*")
    parser.add_argument("--plot-x-column", default="ConnectorDistance")
    parser.add_argument("--plot-rmsd-max", type=float)
    parser.add_argument("--plot-format", action="append", dest="plot_formats")
    parser.add_argument("--plot-dpi", type=int)
    parser.add_argument("--plot-ci", type=float)
    parser.add_argument("--plot-lock-timeout", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.alignment_mode == "fr-ca-kabsch":
        raise ValueError(
            "FR C-alpha Kabsch analysis is implemented by "
            "analyze_resolved_cdr_variation.py because it requires observed-sequence "
            "CDR mapping and coverage-aware QC"
        )
    plot_kwargs = {key: value for key, value in {
        "output_dir": args.plot_out_dir, "metrics": args.plot_metrics, "x_column": args.plot_x_column,
        "rmsd_max": args.plot_rmsd_max, "formats": args.plot_formats, "dpi": args.plot_dpi,
        "ci": args.plot_ci, "plot_lock_timeout": args.plot_lock_timeout,
    }.items() if value is not None}
    run_analysis(
        groups=args.groups, metadata_tsv=args.metadata_tsv, pdb_dir=args.pdb_dir,
        usalign_path=args.usalign_path, out=args.out, member_col=args.member_col,
        group_id_col=args.group_id_col, metadata_id_col=args.metadata_id_col,
        structure_path_col=args.structure_path_col, target_region_cols=args.target_region_cols,
        emb=args.emb, embedding_id_col=args.embedding_id_col,
        embedding_duplicate_policy=args.embedding_duplicate_policy, connector=args.connector,
        cache_dir=args.cache_dir, resume=args.resume, run_id=args.run_id,
        export_aligned_dir=args.export_aligned_dir, timeout=args.timeout, workers=args.workers,
        executor=args.executor, max_in_flight=args.max_in_flight, lock_ttl=args.lock_ttl,
        plot_after_complete=args.plot_after_complete, plot_kwargs=plot_kwargs,
    )
    parameter_record = {
        "alignment_mode": args.alignment_mode,
        "cdr_definition": args.cdr_definition,
        "qc_min_coverage": args.qc_min_coverage,
        "bootstrap_resamples": args.bootstrap_resamples,
        "permutation_resamples": args.permutation_resamples,
        "random_seed": args.random_seed,
        "connector_normalization_residues": args.connector,
    }
    Path(args.out).with_suffix(".parameters.json").write_text(
        json.dumps(parameter_record, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
