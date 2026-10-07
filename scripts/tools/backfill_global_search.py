#!/usr/bin/env python3
"""Backfill Foldseek / MMseqs global-search tables for table-mode grafting.

Table mode disables ``global_search``, so each task folder only has connector
embedding similarities. This script writes the notebook-required files:

  <task>/similarity/<pdb_id>_foldseek_output.tsv
  <task>/similarity/<pdb_id>_mmseqs_output.tsv

Default databases match the id60 cluster-representative grafting run.
A relaxed E-value (``-e 10000``) is used so all 35 FRs are retained for qident
plots; grafting itself is not re-run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

FINAL_VERSION_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = FINAL_VERSION_ROOT / "data" / "batch_graft_generation_id60_table"
DEFAULT_PDB_DB = (
    FINAL_VERSION_ROOT
    / "data"
    / "INDI_database"
    / "mmseqs_cluster_structure"
    / "id60"
    / "cluster_rep_structures"
)
DEFAULT_SEQS = (
    FINAL_VERSION_ROOT
    / "data"
    / "INDI_database"
    / "mmseqs_cluster_structure"
    / "id60"
    / "cluster_rep_seq.fasta"
)
DEFAULT_ENV = "/data1/dhuang/miniconda3/envs/foldseek/bin/"

FOLDSEEK_FORMAT = (
    "query,target,fident,alnlen,nident,mismatch,gapopen,"
    "qstart,qend,tstart,tend,evalue,qtmscore,bits,qaln,taln,tseq"
)
MMSEQS_FORMAT = (
    "query,target,fident,alnlen,nident,mismatch,gapopen,"
    "qstart,qend,tstart,tend,evalue,bits,qaln,taln,tseq"
)

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def list_task_dirs(root: Path) -> list[Path]:
    tasks = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / f"{path.name}.pdb").exists():
            tasks.append(path)
    return tasks


def run_command(command: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"$ {command}\n")
        log_file.flush()
        result = subprocess.run(
            command,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}")


def write_query_fasta(query_pdb: Path, fasta_path: Path) -> None:
    import readPDBSeq

    seq = readPDBSeq.get_sequence_from_pdb(str(query_pdb))
    fasta_path.write_text(f">{query_pdb.stem}\n{seq}\n")


def search_one(
    prefix: str,
    env: str,
    query: Path,
    db_path: Path,
    output_tsv: Path,
    tmp_dir: Path,
    cpu_num: int,
    evalue: float,
    max_seqs: int,
    log_path: Path,
) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    binary = os.path.join(env, prefix)
    if prefix == "foldseek":
        fmt = FOLDSEEK_FORMAT
        query_arg = str(query)
    else:
        fmt = MMSEQS_FORMAT
        fasta_path = output_tsv.parent / "query.fasta"
        write_query_fasta(query, fasta_path)
        query_arg = str(fasta_path)

    command = (
        f"{binary} easy-search {query_arg} {db_path} {output_tsv} {tmp_dir} "
        f"--max-seqs {max_seqs} --remove-tmp-files 1 --threads {cpu_num} "
        f"-e {evalue} --format-mode 4 --format-output '{fmt}'"
    )
    run_command(command, log_path)

    df = pd.read_csv(output_tsv, sep="\t")
    df.columns = [col.strip().lower() for col in df.columns]
    required = {"target", "nident", "bits", "tseq"}
    if prefix == "foldseek":
        required.add("qtmscore")
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{output_tsv} missing columns: {sorted(missing)}")
    df["target"] = df["target"].astype(str).str.replace(r"\.pdb$", "", regex=True)
    df = df.drop_duplicates(subset="target", keep="first")
    df.to_csv(output_tsv, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Foldseek/MMseqs TSV files for table-mode grafting outputs."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Batch grafting output folder")
    parser.add_argument("--pdb-db", default=str(DEFAULT_PDB_DB), help="Foldseek target PDB folder")
    parser.add_argument("--seqs-file", default=str(DEFAULT_SEQS), help="MMseqs target FASTA")
    parser.add_argument("--env", default=DEFAULT_ENV, help="Directory containing foldseek and mmseqs binaries")
    parser.add_argument("--cpu", type=int, default=8, help="Threads per search")
    parser.add_argument("--evalue", type=float, default=10000.0, help="Relaxed E-value to keep all FRs")
    parser.add_argument("--max-seqs", type=int, default=1000, help="Max hits per query")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if output TSV already exists")
    parser.add_argument("--skip-foldseek", action="store_true")
    parser.add_argument("--skip-mmseqs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    pdb_db = Path(args.pdb_db).resolve()
    seqs_file = Path(args.seqs_file).resolve()
    env = args.env if args.env.endswith("/") else args.env + "/"

    if not input_dir.is_dir():
        print(f"[ERROR] Input folder does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not args.skip_foldseek and not pdb_db.is_dir():
        print(f"[ERROR] Foldseek PDB database folder does not exist: {pdb_db}", file=sys.stderr)
        return 1
    if not args.skip_mmseqs and not seqs_file.is_file():
        print(f"[ERROR] MMseqs FASTA does not exist: {seqs_file}", file=sys.stderr)
        return 1

    tasks = list_task_dirs(input_dir)
    if not tasks:
        print(f"[ERROR] No task folders with <id>.pdb found in {input_dir}", file=sys.stderr)
        return 1

    n_ok = n_skip = n_fail = 0
    for task_dir in tasks:
        pdb_id = task_dir.name
        query_pdb = task_dir / f"{pdb_id}.pdb"
        similarity_dir = task_dir / "similarity"
        similarity_dir.mkdir(parents=True, exist_ok=True)
        log_path = similarity_dir / "global_search.log"
        tmp_dir = similarity_dir / "search_tmp"
        try:
            if not args.skip_foldseek:
                out_tsv = similarity_dir / f"{pdb_id}_foldseek_output.tsv"
                if out_tsv.exists() and not args.overwrite:
                    print(f"[SKIP] {pdb_id} foldseek")
                    n_skip += 1
                else:
                    search_one(
                        "foldseek", env, query_pdb, pdb_db, out_tsv, tmp_dir,
                        args.cpu, args.evalue, args.max_seqs, log_path,
                    )
                    print(f"[OK] {pdb_id} foldseek -> {out_tsv.name} ({sum(1 for _ in open(out_tsv)) - 1} hits)")
                    n_ok += 1
            if not args.skip_mmseqs:
                out_tsv = similarity_dir / f"{pdb_id}_mmseqs_output.tsv"
                if out_tsv.exists() and not args.overwrite:
                    print(f"[SKIP] {pdb_id} mmseqs")
                    n_skip += 1
                else:
                    search_one(
                        "mmseqs", env, query_pdb, seqs_file, out_tsv, tmp_dir,
                        args.cpu, args.evalue, args.max_seqs, log_path,
                    )
                    print(f"[OK] {pdb_id} mmseqs -> {out_tsv.name} ({sum(1 for _ in open(out_tsv)) - 1} hits)")
                    n_ok += 1
        except Exception as exc:
            print(f"[ERROR] {pdb_id}: {exc}", file=sys.stderr)
            n_fail += 1

    print(f"Done. ok={n_ok} skipped={n_skip} failed={n_fail} tasks={len(tasks)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    os.chdir(FINAL_VERSION_ROOT)
    raise SystemExit(main())
