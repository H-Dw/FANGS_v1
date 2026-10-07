#!/usr/bin/env python3
"""Summarize FR usage and filter_best retention for batch grafting outputs.

For each grafting task (query / donor PDB), report:
  - how many candidate FRs were used
  - how many generated graftings and unique FRs exist
  - how many rows / unique FRs remain in extract_distance/filter_best
  - which candidate FRs were filtered out, and why

Default input is ``data/batch_graft_generation_id60_table``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

FINAL_VERSION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = FINAL_VERSION_ROOT / "data" / "batch_graft_generation_id60_table"
DEFAULT_OUTPUT = DEFAULT_INPUT / "analysis"
EMBE_LIST = ("raw_embeddings", "pre_q_embeddings", "ca_distance")
STAGE_FILES = {
    "full": "extract_distance/full/grafted_{embe}.tsv",
    "filter": "extract_distance/filter/grafted_{embe}.tsv",
    "filter_best": "extract_distance/filter_best/grafted_{embe}.tsv",
}


def pdb_id_from_path(path: str) -> str:
    return Path(str(path)).stem


def split_donor_fr(combined_id: str, donor: str) -> str:
    prefix = f"{donor}_"
    if str(combined_id).startswith(prefix):
        return str(combined_id)[len(prefix):]
    if "_" in str(combined_id):
        return str(combined_id).split("_", 1)[1]
    return str(combined_id)


def list_task_dirs(root: Path) -> list[Path]:
    tasks = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "temp_generation" / "temp_input.tsv").exists() or (
            path / "temp_generation" / "all_info" / "all_generation.tsv"
        ).exists():
            tasks.append(path)
    return tasks


def read_candidate_frs(task_dir: Path, donor: str) -> list[str]:
    input_tsv = task_dir / "temp_generation" / "temp_input.tsv"
    if not input_tsv.exists():
        table_tsv = task_dir / "temp_generation" / "temp_input_table.tsv"
        if not table_tsv.exists():
            return []
        df = pd.read_csv(table_tsv, sep="\t")
        return [
            pdb_id_from_path(value)
            for value in df["pdb_path"].astype(str)
            if pdb_id_from_path(value) != donor
        ]
    df = pd.read_csv(input_tsv, sep="\t")
    frs = []
    seen = set()
    for value in df["pdb_path"].astype(str):
        fr_id = pdb_id_from_path(value)
        if fr_id == donor or fr_id in seen:
            continue
        seen.add(fr_id)
        frs.append(fr_id)
    return frs


def unique_ids(path: Path, column: str) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t")
    if column not in df.columns:
        return []
    return [str(x) for x in df[column].dropna().astype(str).unique().tolist()]


def n_rows(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path, sep="\t")
    return len(df)


def extract_frs(path: Path, donor: str) -> list[str]:
    combined = unique_ids(path, "PDB_ID")
    return [split_donor_fr(value, donor) for value in combined]


def classify_fr(
    fr_id: str,
    generated: set[str],
    full_frs: set[str],
    filter_frs: set[str],
    filter_best_frs: set[str],
) -> str:
    if fr_id in filter_best_frs:
        return "kept_in_filter_best"
    if fr_id in filter_frs:
        return "in_filter_not_best"
    if fr_id in full_frs:
        return "filtered_by_change"
    if fr_id in generated:
        return "generated_missing_embeddings"
    return "not_generated"


def summarize_task(task_dir: Path, embe_list: Iterable[str]) -> tuple[dict, list[dict]]:
    donor = task_dir.name
    candidates = read_candidate_frs(task_dir, donor)
    gen_tsv = task_dir / "temp_generation" / "all_info" / "all_generation.tsv"
    generated_frs = unique_ids(gen_tsv, "PDB_ID")
    n_generated_samples = n_rows(gen_tsv)

    row = {
        "query_id": donor,
        "n_candidate_fr": len(candidates),
        "candidate_frs": ",".join(candidates),
        "n_generated_fr": len(generated_frs),
        "n_generated_samples": n_generated_samples,
        "has_all_generation": gen_tsv.exists(),
    }
    details: list[dict] = []
    generated_set = set(generated_frs)
    candidate_set = set(candidates)

    for embe in embe_list:
        full_path = task_dir / STAGE_FILES["full"].format(embe=embe)
        filt_path = task_dir / STAGE_FILES["filter"].format(embe=embe)
        best_path = task_dir / STAGE_FILES["filter_best"].format(embe=embe)
        full_frs = extract_frs(full_path, donor)
        filter_frs = extract_frs(filt_path, donor)
        best_frs = extract_frs(best_path, donor)
        prefix = embe.replace("_embeddings", "").replace("ca_distance", "ca")
        row[f"n_full_samples_{prefix}"] = n_rows(full_path)
        row[f"n_full_fr_{prefix}"] = len(full_frs)
        row[f"n_filter_samples_{prefix}"] = n_rows(filt_path)
        row[f"n_filter_fr_{prefix}"] = len(filter_frs)
        row[f"n_filter_best_{prefix}"] = n_rows(best_path)
        row[f"n_filter_best_fr_{prefix}"] = len(best_frs)
        row[f"filter_best_frs_{prefix}"] = ",".join(sorted(best_frs))
        filtered_out = sorted(candidate_set - set(best_frs))
        row[f"n_filtered_out_fr_{prefix}"] = len(filtered_out)
        row[f"filtered_out_frs_{prefix}"] = ",".join(filtered_out)

        full_set = set(full_frs)
        filter_set = set(filter_frs)
        best_set = set(best_frs)
        all_frs = sorted(candidate_set | generated_set | full_set | filter_set | best_set)
        for fr_id in all_frs:
            details.append(
                {
                    "query_id": donor,
                    "fr_id": fr_id,
                    "embe": embe,
                    "is_candidate": fr_id in candidate_set,
                    "is_generated": fr_id in generated_set,
                    "in_full": fr_id in full_set,
                    "in_filter": fr_id in filter_set,
                    "in_filter_best": fr_id in best_set,
                    "status": classify_fr(
                        fr_id, generated_set, full_set, filter_set, best_set
                    ),
                }
            )

    return row, details


def write_overview(summary_df: pd.DataFrame, embe_list: Iterable[str], path: Path) -> None:
    lines = []
    n_tasks = len(summary_df)
    lines.append(f"Grafting tasks: {n_tasks}")
    if n_tasks == 0:
        path.write_text("\n".join(lines) + "\n")
        return
    lines.append(
        "Candidate FRs per task: "
        f"min={int(summary_df['n_candidate_fr'].min())}, "
        f"median={summary_df['n_candidate_fr'].median():.0f}, "
        f"max={int(summary_df['n_candidate_fr'].max())}, "
        f"mean={summary_df['n_candidate_fr'].mean():.2f}"
    )
    lines.append(
        "Generated samples per task: "
        f"min={int(summary_df['n_generated_samples'].min())}, "
        f"median={summary_df['n_generated_samples'].median():.0f}, "
        f"max={int(summary_df['n_generated_samples'].max())}"
    )
    for embe in embe_list:
        prefix = embe.replace("_embeddings", "").replace("ca_distance", "ca")
        best_col = f"n_filter_best_{prefix}"
        out_col = f"n_filtered_out_fr_{prefix}"
        if best_col not in summary_df.columns:
            continue
        lines.append(
            f"[{embe}] filter_best graftings: "
            f"total={int(summary_df[best_col].sum())}, "
            f"per-task min/median/max="
            f"{int(summary_df[best_col].min())}/"
            f"{summary_df[best_col].median():.0f}/"
            f"{int(summary_df[best_col].max())}; "
            f"FRs filtered out per task mean="
            f"{summary_df[out_col].mean():.2f}"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize candidate FR counts and filter_best retention."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Batch grafting output folder (one subdirectory per query PDB)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Folder for summary TSV / overview files",
    )
    parser.add_argument(
        "--embe-list",
        nargs="+",
        default=list(EMBE_LIST),
        help="Embedding types whose extract_distance tables are counted",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"[ERROR] Input folder does not exist: {input_dir}", file=sys.stderr)
        return 1

    tasks = list_task_dirs(input_dir)
    if not tasks:
        print(f"[ERROR] No grafting task folders found in {input_dir}", file=sys.stderr)
        return 1

    summary_rows = []
    detail_rows = []
    for task_dir in tasks:
        row, details = summarize_task(task_dir, args.embe_list)
        summary_rows.append(row)
        detail_rows.extend(details)
        print(
            f"{row['query_id']}: candidate FR={row['n_candidate_fr']}, "
            f"generated samples={row['n_generated_samples']}, "
            f"filter_best(raw)={row.get('n_filter_best_raw', 'NA')}"
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    summary_path = output_dir / "task_fr_summary.tsv"
    detail_path = output_dir / "fr_status.tsv"
    filtered_path = output_dir / "filtered_fr_by_task.tsv"
    overview_path = output_dir / "overview.txt"

    summary_df.to_csv(summary_path, sep="\t", index=False)
    detail_df.to_csv(detail_path, sep="\t", index=False)

    filtered_df = detail_df[
        (detail_df["is_candidate"]) & (~detail_df["in_filter_best"])
    ].copy()
    filtered_df.to_csv(filtered_path, sep="\t", index=False)
    write_overview(summary_df, args.embe_list, overview_path)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {filtered_path}")
    print(f"Wrote {overview_path}")
    print(overview_path.read_text().rstrip())
    return 0


if __name__ == "__main__":
    os.chdir(FINAL_VERSION_ROOT)
    raise SystemExit(main())
