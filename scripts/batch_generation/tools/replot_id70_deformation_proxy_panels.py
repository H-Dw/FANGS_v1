#!/usr/bin/env python3
"""Replot the consolidated id70 deformation-risk sensitivity panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_id70_virtual_grafting_sensitivity import (
    plot_threshold_outcomes,
    plot_within_donor,
)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def ensure_absolute_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["abs_rho"] = result["rho"].abs()
    if "two_way_abs_ci_low" not in result.columns:
        crosses_zero = (result["two_way_ci_low"] <= 0) & (result["two_way_ci_high"] >= 0)
        result["two_way_abs_ci_low"] = 0.0
        non_crossing = ~crosses_zero
        result.loc[non_crossing, "two_way_abs_ci_low"] = result.loc[
            non_crossing, ["two_way_ci_low", "two_way_ci_high"]
        ].abs().min(axis=1)
        result["two_way_abs_ci_high"] = result[
            ["two_way_ci_low", "two_way_ci_high"]
        ].abs().max(axis=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(args.table_dir / "id70_threshold_curves.tsv", sep="\t")
    anchors = pd.read_csv(args.table_dir / "id70_threshold_anchor_statistics.tsv", sep="\t")
    within = pd.read_csv(args.table_dir / "id70_within_donor_cumulative_stability.tsv", sep="\t")
    curves = numeric(curves, ["threshold", "rho", "retained_design_percent"])
    anchors = numeric(anchors, ["rho", "two_way_ci_low", "two_way_ci_high"])
    within = numeric(
        within,
        [
            "within_donor_top_percent",
            "rho",
            "two_way_ci_low",
            "two_way_ci_high",
            "n_cells",
        ],
    )
    anchors = ensure_absolute_columns(anchors)
    within = ensure_absolute_columns(within)

    plot_threshold_outcomes(curves, anchors, args.output_dir, table_dir=args.table_dir)
    plot_within_donor(within, args.output_dir, table_dir=args.table_dir)
    print(
        json.dumps(
            {
                "replotted_panels": 2,
                "output_dir": str(args.output_dir),
                "threshold_panel": "Fig1h_id70_delta_connector_threshold_scan",
                "within_donor_panel": "Fig1i_id70_donor_cumulative_neighborhood",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
