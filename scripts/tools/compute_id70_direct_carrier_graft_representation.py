#!/usr/bin/env python3
"""Compute direct carrier-graft RAW connector representations for id70 designs.

The script processes only the complete sample row selected for each realized
donor-carrier design. It scans each donor's generated RAW-vector table once,
extracts the requested generated rows, and compares them with the matching
native carrier vectors. ESM3 is not rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_RESIDUES = 12
EXPECTED_CONNECTORS = 3
EXPECTED_RESIDUES_PER_CONNECTOR = 4
KEY_COLUMNS = ["Donator", "FR_ID", "Filename"]


@dataclass(frozen=True)
class DonorTask:
    donor: str
    generation_root: str
    selected_records: list[dict]
    part_dir: str
    resume: bool
    tolerance: float


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalize_generated_id(value: str) -> str:
    name = Path(str(value)).name
    if name.endswith(".pdb"):
        name = name[:-4]
    return name


def read_vector_table(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    frame = pd.read_csv(path, sep="\t", dtype={0: str})
    identifiers = frame.iloc[:, 0].astype(str).tolist()
    vectors = frame.iloc[:, 1:].to_numpy(dtype=np.float64, copy=False)
    return identifiers, dict(zip(identifiers, vectors))


def read_single_vector(path: Path, expected_id: str) -> np.ndarray:
    identifiers, vectors = read_vector_table(path)
    if expected_id in vectors:
        return vectors[expected_id]
    if len(identifiers) == 1:
        return vectors[identifiers[0]]
    raise RuntimeError(f"Query vector {expected_id} was not found in {path}")


def scan_selected_generated_vectors(
    path: Path,
    requested_ids: set[str],
) -> tuple[dict[str, np.ndarray], int]:
    found: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        n_columns = header.rstrip("\r\n").count("\t") + 1
        n_features = n_columns - 1
        for line in handle:
            identifier, separator, payload = line.partition("\t")
            if not separator or identifier not in requested_ids:
                continue
            vector = np.fromstring(payload, sep="\t", dtype=np.float64)
            if vector.size != n_features:
                raise RuntimeError(
                    f"Generated vector {identifier} has {vector.size} values, expected {n_features}"
                )
            found[identifier] = vector
    missing = sorted(requested_ids.difference(found))
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Missing {len(missing)} generated vectors in {path}: {preview}")
    return found, path.stat().st_size


def direct_metrics(carrier: np.ndarray, graft: np.ndarray) -> dict[str, float]:
    if carrier.shape != graft.shape:
        raise RuntimeError(f"Vector shape mismatch: carrier {carrier.shape}, graft {graft.shape}")
    if carrier.size % EXPECTED_RESIDUES != 0:
        raise RuntimeError(
            f"RAW vector width {carrier.size} is not divisible by {EXPECTED_RESIDUES} residues"
        )
    if carrier.size % EXPECTED_CONNECTORS != 0:
        raise RuntimeError(
            f"RAW vector width {carrier.size} is not divisible by {EXPECTED_CONNECTORS} connectors"
        )

    difference = carrier - graft
    connector_width = carrier.size // EXPECTED_CONNECTORS
    connector_raw = []
    for connector_index in range(EXPECTED_CONNECTORS):
        start = connector_index * connector_width
        stop = start + connector_width
        connector_raw.append(float(np.linalg.norm(difference[start:stop])))
    overall_raw = float(np.linalg.norm(difference))
    connector_norm = [value / math.sqrt(EXPECTED_RESIDUES_PER_CONNECTOR) for value in connector_raw]
    overall_norm = overall_raw / math.sqrt(EXPECTED_RESIDUES)
    block_identity = math.sqrt(sum(value * value for value in connector_norm) / EXPECTED_CONNECTORS)
    return {
        "direct_carrier_graft_all_raw": overall_raw,
        "direct_carrier_graft_all_raw_norm": overall_norm,
        "direct_carrier_graft_connector1_raw": connector_raw[0],
        "direct_carrier_graft_connector1_raw_norm": connector_norm[0],
        "direct_carrier_graft_connector2_raw": connector_raw[1],
        "direct_carrier_graft_connector2_raw_norm": connector_norm[1],
        "direct_carrier_graft_connector3_raw": connector_raw[2],
        "direct_carrier_graft_connector3_raw_norm": connector_norm[2],
        "block_identity_norm": block_identity,
        "block_identity_abs_error": abs(overall_norm - block_identity),
    }


def validate_part(path: Path, selected: pd.DataFrame, tolerance: float) -> bool:
    if not path.exists():
        return False
    try:
        part = pd.read_csv(path, sep="\t")
    except Exception:
        return False
    expected_keys = set(map(tuple, selected[KEY_COLUMNS].astype(str).to_numpy()))
    observed_keys = set(map(tuple, part[KEY_COLUMNS].astype(str).to_numpy()))
    if len(part) != len(selected) or observed_keys != expected_keys:
        return False
    if part["direct_carrier_graft_all_raw_norm"].isna().any():
        return False
    if float(part["block_identity_abs_error"].max()) > tolerance:
        return False
    return True


def process_donor(task: DonorTask) -> dict:
    started = time.time()
    donor = task.donor
    root = Path(task.generation_root)
    donor_root = root / donor
    part_dir = Path(task.part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"{donor}.tsv"
    selected = pd.DataFrame(task.selected_records)
    selected["generated_id"] = selected["Filename"].map(normalize_generated_id)

    if task.resume and validate_part(part_path, selected, task.tolerance):
        return {
            "donor": donor,
            "rows": int(len(selected)),
            "status": "reused",
            "seconds": time.time() - started,
            "bytes_read": 0,
            "part_path": str(part_path),
        }

    carrier_path = donor_root / "tokenized_db" / "raw_embeddings.tsv"
    graft_path = donor_root / "generation_tokenized_structure" / "raw_embeddings.tsv"
    donor_path = donor_root / "tokenized_input" / "raw_embeddings.tsv"
    for required in (carrier_path, graft_path, donor_path):
        if not required.exists():
            raise FileNotFoundError(required)

    carrier_ids, carrier_vectors = read_vector_table(carrier_path)
    missing_carriers = sorted(set(selected["FR_ID"]).difference(carrier_vectors))
    if missing_carriers:
        preview = ", ".join(missing_carriers[:10])
        raise RuntimeError(f"Missing {len(missing_carriers)} carrier vectors for {donor}: {preview}")
    donor_vector = read_single_vector(donor_path, donor)
    generated_vectors, bytes_read = scan_selected_generated_vectors(
        graft_path,
        set(selected["generated_id"]),
    )

    rows = []
    for record in selected.to_dict("records"):
        carrier = carrier_vectors[str(record["FR_ID"])]
        graft = generated_vectors[str(record["generated_id"])]
        metrics = direct_metrics(carrier, graft)
        recomputed_original = float(np.linalg.norm(donor_vector - carrier) / math.sqrt(EXPECTED_RESIDUES))
        recomputed_grafted = float(np.linalg.norm(donor_vector - graft) / math.sqrt(EXPECTED_RESIDUES))
        original_error = abs(
            recomputed_original - float(record["original_euclidean_all_raw_embeddings_norm"])
        )
        grafted_error = abs(
            recomputed_grafted - float(record["grafted_euclidean_all_raw_embeddings_norm"])
        )
        rows.append(
            {
                "Donator": donor,
                "FR_ID": str(record["FR_ID"]),
                "Filename": str(record["Filename"]),
                "generated_id": str(record["generated_id"]),
                **metrics,
                "recomputed_original_all_raw_norm": recomputed_original,
                "recomputed_grafted_all_raw_norm": recomputed_grafted,
                "original_distance_abs_error": original_error,
                "grafted_distance_abs_error": grafted_error,
            }
        )

    result = pd.DataFrame(rows).sort_values(["Donator", "FR_ID", "Filename"])
    if float(result["block_identity_abs_error"].max()) > task.tolerance:
        raise RuntimeError(f"Connector block identity failed for donor {donor}")
    if float(result["original_distance_abs_error"].max()) > task.tolerance:
        raise RuntimeError(f"Donor-carrier audit failed for donor {donor}")
    if float(result["grafted_distance_abs_error"].max()) > task.tolerance:
        raise RuntimeError(f"Donor-graft audit failed for donor {donor}")

    temporary = part_path.with_suffix(".tmp.tsv")
    result.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, part_path)
    return {
        "donor": donor,
        "rows": int(len(result)),
        "status": "computed",
        "seconds": time.time() - started,
        "bytes_read": int(bytes_read),
        "part_path": str(part_path),
        "carrier_rows": int(len(carrier_ids)),
        "max_block_identity_abs_error": float(result["block_identity_abs_error"].max()),
        "max_original_distance_abs_error": float(result["original_distance_abs_error"].max()),
        "max_grafted_distance_abs_error": float(result["grafted_distance_abs_error"].max()),
    }


def parse_donors(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--selected-designs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--donors", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_dir = args.output_dir / "direct_connector_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_designs, sep="\t")
    required_columns = {
        "Donator",
        "FR_ID",
        "Filename",
        "original_euclidean_all_raw_embeddings_norm",
        "grafted_euclidean_all_raw_embeddings_norm",
    }
    missing_columns = sorted(required_columns.difference(selected.columns))
    if missing_columns:
        raise RuntimeError(f"Selected design table lacks columns: {missing_columns}")
    if selected.duplicated(["Donator", "FR_ID"]).any():
        raise RuntimeError("Selected design table contains duplicate donor-carrier cells")

    donor_filter = parse_donors(args.donors)
    if donor_filter is not None:
        missing_donors = sorted(donor_filter.difference(set(selected["Donator"])))
        if missing_donors:
            raise RuntimeError(f"Requested donors were not found: {missing_donors}")
        selected = selected[selected["Donator"].isin(donor_filter)].copy()
    selected = selected.sort_values(["Donator", "FR_ID", "Filename"]).reset_index(drop=True)

    tasks = []
    for donor, group in selected.groupby("Donator", sort=True):
        tasks.append(
            DonorTask(
                donor=str(donor),
                generation_root=str(args.generation_root),
                selected_records=group[
                    [
                        "Donator",
                        "FR_ID",
                        "Filename",
                        "original_euclidean_all_raw_embeddings_norm",
                        "grafted_euclidean_all_raw_embeddings_norm",
                    ]
                ].to_dict("records"),
                part_dir=str(part_dir),
                resume=bool(args.resume),
                tolerance=float(args.tolerance),
            )
        )

    progress = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_donor, task): task.donor for task in tasks}
        completed = 0
        for future in as_completed(futures):
            item = future.result()
            progress.append(item)
            completed += 1
            print(
                f"[{completed}/{len(tasks)}] {item['donor']} {item['status']} "
                f"rows={item['rows']} seconds={item['seconds']:.1f}",
                flush=True,
            )

    parts = [pd.read_csv(part_dir / f"{donor}.tsv", sep="\t") for donor in sorted(selected["Donator"].unique())]
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["Donator", "FR_ID", "Filename"]).reset_index(drop=True)
    expected_keys = set(map(tuple, selected[KEY_COLUMNS].astype(str).to_numpy()))
    observed_keys = set(map(tuple, result[KEY_COLUMNS].astype(str).to_numpy()))
    if len(result) != len(selected):
        raise RuntimeError(f"Expected {len(selected)} direct rows, observed {len(result)}")
    if result.duplicated(["Donator", "FR_ID"]).any():
        raise RuntimeError("Direct metric table contains duplicate donor-carrier cells")
    if observed_keys != expected_keys:
        raise RuntimeError("Direct metric keys do not match the selected design table")
    numeric_columns = [column for column in result.columns if column not in KEY_COLUMNS + ["generated_id"]]
    if result[numeric_columns].isna().any().any():
        raise RuntimeError("Direct metric table contains missing numeric values")

    output_table = args.output_dir / "id70_direct_carrier_graft_representation_selected.tsv"
    result.to_csv(output_table, sep="\t", index=False)
    progress_frame = pd.DataFrame(progress).sort_values("donor")
    progress_frame.to_csv(args.output_dir / "id70_direct_carrier_graft_progress.tsv", sep="\t", index=False)

    metadata = {
        "generation_root": str(args.generation_root.resolve()),
        "selected_designs": str(args.selected_designs.resolve()),
        "selected_designs_sha256": sha256_file(args.selected_designs),
        "output_table": str(output_table.resolve()),
        "designs": int(len(result)),
        "donors": int(result["Donator"].nunique()),
        "carriers": int(result["FR_ID"].nunique()),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "normalization_overall": "Euclidean difference divided by sqrt(12)",
        "normalization_connector": "Euclidean difference divided by sqrt(4)",
        "max_block_identity_abs_error": float(result["block_identity_abs_error"].max()),
        "max_original_distance_abs_error": float(result["original_distance_abs_error"].max()),
        "max_grafted_distance_abs_error": float(result["grafted_distance_abs_error"].max()),
        "bytes_scanned": int(progress_frame["bytes_read"].sum()),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "id70_direct_carrier_graft_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
