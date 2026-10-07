#!/usr/bin/env python3
"""Build Supplementary Table 4 for id70 representation-stage sensitivity."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BATCH_TOOLS = Path(__file__).resolve().parents[1] / "batch_generation" / "tools"
if str(_BATCH_TOOLS) not in sys.path:
    sys.path.insert(0, str(_BATCH_TOOLS))

from analyze_id70_virtual_grafting_sensitivity import crossed_two_way_bootstrap
from revision_common import benjamini_hochberg, cluster_spearman


KEY_COLUMNS = ["Donator", "FR_ID", "Filename"]
DESCRIPTORS = {
    "RAW": ("raw_embeddings", math.sqrt(12.0)),
    "pre-VQ": ("pre_q_embeddings", math.sqrt(12.0)),
    "Cα": ("ca_distance", math.sqrt(12.0)),
}
STAGES = ("Original", "Grafted")
OUTCOMES = ("cRMSD", "pTM")


def normalize_identifier(value: str) -> str:
    name = Path(str(value)).name
    return name[:-4] if name.endswith(".pdb") else name


def choose_identifier_column(frame: pd.DataFrame) -> str:
    for candidate in ("Filename", "PDB_ID", "ID", "name"):
        if candidate in frame.columns:
            return candidate
    return str(frame.columns[0])


def choose_distance_column(frame: pd.DataFrame, stage: str, descriptor_token: str) -> tuple[str, bool]:
    lowered = {column: column.lower() for column in frame.columns}
    candidates = [
        column
        for column, low in lowered.items()
        if "euclidean_all" in low and descriptor_token in low and stage.lower() in low
    ]
    if not candidates:
        candidates = [
            column
            for column, low in lowered.items()
            if "euclidean_all" in low and descriptor_token in low
        ]
    if not candidates:
        raise RuntimeError(
            f"No whole-connector {descriptor_token} distance column was found in {list(frame.columns)}"
        )
    candidates.sort(key=lambda column: ("norm" not in column.lower(), len(column)))
    column = candidates[0]
    return column, "norm" in column.lower()


def read_stage_values(
    donor_root: Path,
    stage: str,
    descriptor_token: str,
    divisor: float,
) -> dict[str, float]:
    file_stage = stage.lower()
    path = donor_root / "extract_distance" / "full" / f"{file_stage}_{descriptor_token}.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t")
    identifier_column = choose_identifier_column(frame)
    distance_column, already_normalized = choose_distance_column(frame, file_stage, descriptor_token)
    values = pd.to_numeric(frame[distance_column], errors="coerce")
    if not already_normalized:
        values = values / divisor
    identifiers = frame[identifier_column].astype(str).map(normalize_identifier)
    if identifiers.duplicated().any():
        duplicated = identifiers[identifiers.duplicated()].iloc[0]
        raise RuntimeError(f"Duplicate identifier {duplicated} in {path}")
    return dict(zip(identifiers, values))


def collect_selected_values(selected: pd.DataFrame, generation_root: Path) -> pd.DataFrame:
    rows = []
    for donor, group in selected.groupby("Donator", sort=True):
        donor_root = generation_root / str(donor)
        lookup = {}
        for descriptor, (token, divisor) in DESCRIPTORS.items():
            for stage in STAGES:
                lookup[(descriptor, stage)] = read_stage_values(donor_root, stage, token, divisor)
        for record in group.to_dict("records"):
            graft_id = normalize_identifier(record["Filename"])
            for descriptor in DESCRIPTORS:
                original = lookup[(descriptor, "Original")].get(graft_id)
                grafted = lookup[(descriptor, "Grafted")].get(graft_id)
                if original is None or grafted is None:
                    raise RuntimeError(
                        f"Missing {descriptor} values for donor {donor}, carrier {record['FR_ID']}, graft {graft_id}"
                    )
                rows.append(
                    {
                        **{column: record[column] for column in KEY_COLUMNS},
                        "descriptor": descriptor,
                        "stage": "Original",
                        "representation": original,
                        "cRMSD": record["cRMSD"],
                        "pTM": record["pTM"],
                    }
                )
                rows.append(
                    {
                        **{column: record[column] for column in KEY_COLUMNS},
                        "descriptor": descriptor,
                        "stage": "Grafted",
                        "representation": grafted,
                        "cRMSD": record["cRMSD"],
                        "pTM": record["pTM"],
                    }
                )
    result = pd.DataFrame(rows)
    expected = len(selected) * len(DESCRIPTORS) * len(STAGES)
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} descriptor-stage rows, observed {len(result)}")
    return result


def summarize(values: pd.DataFrame, n_resamples: int, seed: int) -> pd.DataFrame:
    rows = []
    index = 0
    for descriptor in DESCRIPTORS:
        for stage in STAGES:
            subset = values[(values["descriptor"] == descriptor) & (values["stage"] == stage)].copy()
            for outcome in OUTCOMES:
                donor = cluster_spearman(
                    subset,
                    "representation",
                    outcome,
                    "Donator",
                    n_resamples=n_resamples,
                    seed=seed + index * 10,
                )
                carrier = cluster_spearman(
                    subset,
                    "representation",
                    outcome,
                    "FR_ID",
                    n_resamples=n_resamples,
                    seed=seed + index * 10 + 1,
                )
                crossed = crossed_two_way_bootstrap(
                    subset,
                    [("target", "representation", outcome)],
                    n_resamples,
                    seed + index * 10 + 2,
                )["target"]
                rows.append(
                    {
                        "descriptor": descriptor,
                        "stage": stage,
                        "outcome": outcome,
                        "rho": donor["rho"],
                        "abs_rho": donor["abs_rho"],
                        "donor_ci_low": donor["ci_low"],
                        "donor_ci_high": donor["ci_high"],
                        "donor_abs_ci_low": donor["abs_ci_low"],
                        "donor_abs_ci_high": donor["abs_ci_high"],
                        "carrier_ci_low": carrier["ci_low"],
                        "carrier_ci_high": carrier["ci_high"],
                        "carrier_abs_ci_low": carrier["abs_ci_low"],
                        "carrier_abs_ci_high": carrier["abs_ci_high"],
                        "two_way_ci_low": crossed["two_way_ci_low"],
                        "two_way_ci_high": crossed["two_way_ci_high"],
                        "two_way_abs_ci_low": crossed["two_way_abs_ci_low"],
                        "two_way_abs_ci_high": crossed["two_way_abs_ci_high"],
                        "donor_permutation_p": donor["cluster_randomization_p"],
                        "raw_spearman_p": donor["raw_spearman_p"],
                        "n_cells": len(subset),
                        "n_donors": subset["Donator"].nunique(),
                        "n_carriers": subset["FR_ID"].nunique(),
                        "normalization": "Euclidean difference divided by sqrt(12)",
                    }
                )
                index += 1
    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result["donor_permutation_p"].tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--selected-designs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected_designs, sep="\t")
    if selected.duplicated(["Donator", "FR_ID"]).any():
        raise RuntimeError("Selected design table contains duplicate donor-carrier cells")
    if len(selected) != 33856:
        raise RuntimeError(f"Expected 33856 selected design cells, observed {len(selected)}")
    values = collect_selected_values(selected, args.generation_root)
    values.to_csv(
        args.output_dir / "Supplementary_Table_4_id70_representation_stage_source.tsv",
        sep="\t",
        index=False,
    )
    result = summarize(values, args.n_resamples, args.seed)
    result.to_csv(
        args.output_dir / "Supplementary_Table_4_id70_representation_stage_correlations.tsv",
        sep="\t",
        index=False,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
