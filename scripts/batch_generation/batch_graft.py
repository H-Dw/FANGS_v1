#!/usr/bin/env python3
"""Batch CDR grafting with one persistent ESM3 model per GPU."""

import argparse
import logging
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

FINAL_VERSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BATCH_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(FINAL_VERSION_ROOT, 'scripts', 'tools')
CODE_DIR = os.path.abspath(os.path.join(FINAL_VERSION_ROOT, '..', 'code'))

for path in (FINAL_VERSION_ROOT, BATCH_DIR, TOOLS_DIR, CODE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

SUCCESS_MARKER = os.path.join('temp_generation', 'all_info', 'all_generation.tsv')
FINISHED_PATTERN = re.compile(r"Finished processing PDB ID:\s*(\S+)")
FAILED_PATTERN = re.compile(r"Failed processing PDB ID:\s*(\S+)")


class ThreadLocalStream:
    """Route writes to a stream assigned to the current worker thread."""

    def __init__(self, default_stream):
        self.default_stream = default_stream
        self.local = threading.local()

    def set_stream(self, stream) -> None:
        self.local.stream = stream

    def clear_stream(self) -> None:
        if hasattr(self.local, 'stream'):
            del self.local.stream

    def write(self, data):
        return getattr(self.local, 'stream', self.default_stream).write(data)

    def flush(self) -> None:
        getattr(self.local, 'stream', self.default_stream).flush()

    def isatty(self) -> bool:
        stream = getattr(self.local, 'stream', self.default_stream)
        return bool(getattr(stream, 'isatty', lambda: False)())


@dataclass
class BatchConfig:
    pdb_db: str
    seqs_file: str
    tokenized_folder: Optional[str]
    cdr_table: str
    output_path: str
    structure_sample: int
    structure_batch_size: int
    top_n: int
    cpu_num: int
    workers: int
    select_mode: str
    s_type: int
    len_limit: int
    s_temperature: float
    distance_limit: float
    seed: Optional[int]
    dupl_pdb: bool
    side_chain_max: int
    bf16: bool


@dataclass
class TaskResult:
    pdb_id: str
    success: bool
    message: str


def build_cdr_file(pdb_id: str, cdr_table: str, output_path: str) -> str:
    cdr_df = pd.read_csv(cdr_table, sep='\t')
    target_row = cdr_df[cdr_df['PDBChain'] == pdb_id]
    if target_row.empty:
        raise ValueError(f"No entry for PDBChain={pdb_id} in {cdr_table}")
    if len(target_row) > 1:
        raise ValueError(f"Multiple entries for PDBChain={pdb_id} in {cdr_table}")

    output_info = target_row[['Sequence', 'CDR1', 'CDR2', 'CDR3', 'PDBChain']]
    os.makedirs(output_path, exist_ok=True)
    output_filename = os.path.join(output_path, f'{pdb_id}.tsv')
    output_info.to_csv(
        output_filename,
        sep='\t',
        index=False,
        header=['Sequence', 'CDR1', 'CDR2', 'CDR3', 'PDBChain'],
    )
    return output_filename


def validate_inputs(pdb_id: str, pdb_db: str, cdr_table: str, tokenized_folder: Optional[str]) -> None:
    pdb_path = os.path.join(pdb_db, f'{pdb_id}.pdb')
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    cdr_df = pd.read_csv(cdr_table, sep='\t')
    matches = cdr_df[cdr_df['PDBChain'] == pdb_id]
    if matches.empty:
        raise ValueError(f"No CDR entry for PDBChain={pdb_id}")
    if len(matches) > 1:
        raise ValueError(f"Multiple CDR entries for PDBChain={pdb_id}")

    if tokenized_folder is not None:
        for name in ('raw_embeddings.tsv', 'pre_q_embeddings.tsv', 'ca_distance.tsv'):
            path = os.path.join(tokenized_folder, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing tokenized file: {path}")


def is_graft_successful(pdb_folder: str) -> bool:
    return os.path.exists(os.path.join(pdb_folder, SUCCESS_MARKER))


def per_task_cpu(cpu_num: int, workers: int) -> int:
    return max(1, cpu_num // max(1, workers))


def prepare_table_tokenized_folder(
    cdr_table: str, tokenized_folder: Optional[str], output_path: str
) -> Optional[str]:
    """Create a small connector database containing only rows from cdr_table."""
    if tokenized_folder is None:
        return None

    ids = set(pd.read_csv(cdr_table, sep='\t', usecols=['PDBChain'], dtype=str)['PDBChain'])
    table_folder = os.path.join(output_path, '_table_tokenized')
    os.makedirs(table_folder, exist_ok=True)

    for name in ('raw_embeddings.tsv', 'pre_q_embeddings.tsv', 'ca_distance.tsv'):
        source = os.path.join(tokenized_folder, name)
        target = os.path.join(table_folder, name)
        if not os.path.exists(source):
            raise FileNotFoundError(f"Missing tokenized file: {source}")

        selected_chunks = []
        for chunk in pd.read_csv(source, sep='\t', dtype={'ID': str}, chunksize=1000):
            selected = chunk[chunk['ID'].isin(ids)]
            if not selected.empty:
                selected_chunks.append(selected)
        if not selected_chunks:
            raise ValueError(f"None of the table PDBChain IDs were found in {source}")

        selected_df = pd.concat(selected_chunks, ignore_index=True)
        selected_df = selected_df.drop_duplicates(subset='ID', keep='first')
        missing_ids = ids - set(selected_df['ID'])
        if missing_ids:
            raise ValueError(f"Missing tokenized rows in {source}: {sorted(missing_ids)}")
        selected_df.to_csv(target, sep='\t', index=False)

    return table_folder


def process_pdb_task(
    pdb_id: str,
    config: BatchConfig,
    model,
    model_lock: threading.Lock,
) -> TaskResult:
    pdb_folder = os.path.join(config.output_path, pdb_id)
    os.makedirs(pdb_folder, exist_ok=True)

    validate_inputs(pdb_id, config.pdb_db, config.cdr_table, config.tokenized_folder)

    ori_pdb_filename = os.path.join(config.pdb_db, f'{pdb_id}.pdb')
    pdb_filename = os.path.join(pdb_folder, f'{pdb_id}.pdb')
    shutil.copy(ori_pdb_filename, pdb_filename)
    cdr_filename = build_cdr_file(pdb_id=pdb_id, cdr_table=config.cdr_table, output_path=pdb_folder)

    from scripts.grafting_generation.graft_CDR_connector import main as graft_main

    task_cpu = per_task_cpu(config.cpu_num, config.workers)
    graf_log_file = os.path.join(pdb_folder, 'graft_process.log')
    with open(graf_log_file, 'w', encoding='utf-8') as log_file:
        stdout_router = sys.stdout
        stderr_router = sys.stderr
        if not isinstance(stdout_router, ThreadLocalStream) or not isinstance(stderr_router, ThreadLocalStream):
            raise RuntimeError('GPU worker did not initialize thread-local log streams')
        stdout_router.set_stream(log_file)
        stderr_router.set_stream(log_file)
        try:
            graft_main(
                Nb_pdb_path=pdb_filename,
                Nb_CDR_file=cdr_filename,
                FR_pdb_folder=config.pdb_db,
                FR_CDR_file=config.cdr_table,
                output_path=pdb_folder,
                ag_pdb_path='',
                structure_sample=config.structure_sample,
                gpu='',
                cpu_num=task_cpu,
                non_generation=False,
                select_mode=config.select_mode,
                s_type=config.s_type,
                top_n=config.top_n,
                len_limit=config.len_limit,
                seqs_file=config.seqs_file,
                tokenized_folder=config.tokenized_folder,
                global_search=config.select_mode in ('all', 'foldseek', 'mmseqs'),
                alignment=False,
                dupl_pdb=config.dupl_pdb,
                s_temperature=config.s_temperature,
                distance_limit=config.distance_limit,
                seed=config.seed,
                model=model,
                model_lock=model_lock,
                gpu_managed=True,
                output_stream=log_file,
                tmp_subdir=os.path.join(pdb_folder, 'similarity', 'search_tmp'),
                raise_on_error=True,
                side_chain_max=config.side_chain_max,
                structure_batch_size=config.structure_batch_size,
                bf16=config.bf16,
            )
        finally:
            stdout_router.clear_stream()
            stderr_router.clear_stream()

    if not is_graft_successful(pdb_folder):
        raise RuntimeError(f"Missing expected output: {SUCCESS_MARKER}")

    return TaskResult(pdb_id=pdb_id, success=True, message='ok')


def _worker_thread(
    task_queue: queue.Queue,
    result_queue: queue.Queue,
    config: BatchConfig,
    model,
    model_lock: threading.Lock,
) -> None:
    while True:
        try:
            pdb_id = task_queue.get_nowait()
        except queue.Empty:
            return

        try:
            result = process_pdb_task(pdb_id, config, model, model_lock)
        except Exception as exc:
            result = TaskResult(pdb_id=pdb_id, success=False, message=str(exc))
        result_queue.put(result)
        task_queue.task_done()


def gpu_worker_entry(gpu_id: str, task_ids: Sequence[str], config: BatchConfig, result_queue: mp.Queue) -> None:
    if gpu_id != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    import torch
    from esm.models.esm3 import ESM3

    from scripts.grafting_generation.graft_CDR_connector import set_cpu

    set_cpu(per_task_cpu(config.cpu_num, config.workers))

    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu_id != '' else 'cpu')
    print(f"[GPU {gpu_id or 'CPU'}] Loading ESM3 once on {device}...", flush=True)
    model = ESM3.from_pretrained('esm3_sm_open_v1').to(device)
    if config.bf16 and device.type == 'cuda':
        model = model.to(torch.bfloat16)
    model.eval()
    print(f"[GPU {gpu_id or 'CPU'}] ESM3 loaded (bf16={config.bf16 and device.type == 'cuda'}).", flush=True)

    model_lock = threading.Lock()
    sys.stdout = ThreadLocalStream(sys.stdout)
    sys.stderr = ThreadLocalStream(sys.stderr)
    task_queue: queue.Queue = queue.Queue()
    for pdb_id in task_ids:
        task_queue.put(pdb_id)

    local_result_queue: queue.Queue = queue.Queue()
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [
            executor.submit(
                _worker_thread,
                task_queue,
                local_result_queue,
                config,
                model,
                model_lock,
            )
            for _ in range(config.workers)
        ]
        for future in futures:
            future.result()

    while not local_result_queue.empty():
        result_queue.put(local_result_queue.get())


def parse_gpu_list(gpus_arg: Optional[str], legacy_gpu: str) -> List[str]:
    if gpus_arg:
        return [item.strip() for item in gpus_arg.split(',') if item.strip()]
    if legacy_gpu != '':
        return [legacy_gpu]
    return ['']


def load_finished_ids(log_file: str) -> Tuple[set, set]:
    finished_ids = set()
    failed_ids = set()
    if not os.path.exists(log_file):
        return finished_ids, failed_ids

    with open(log_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            finished_match = FINISHED_PATTERN.search(line)
            if finished_match:
                finished_ids.add(finished_match.group(1))
            failed_match = FAILED_PATTERN.search(line)
            if failed_match:
                failed_ids.add(failed_match.group(1))
    return finished_ids, failed_ids


def distribute_tasks(task_ids: Sequence[str], gpu_ids: Sequence[str]) -> List[List[str]]:
    buckets = [[] for _ in gpu_ids]
    for index, pdb_id in enumerate(task_ids):
        buckets[index % len(gpu_ids)].append(pdb_id)
    return buckets


def run_batch(
    pdb_list_file: str,
    pdb_db: str,
    seqs_file: str,
    tokenized_folder: Optional[str],
    cdr_table: str,
    output_path: str,
    gpu_ids: Sequence[str],
    structure_sample: int,
    structure_batch_size: int,
    top_n: int,
    cpu_num: int,
    workers: int,
    select_mode: str,
    s_type: int,
    len_limit: int,
    s_temperature: float,
    distance_limit: float,
    seed: Optional[int],
    dupl_pdb: bool,
    side_chain_max: int,
    bf16: bool,
) -> None:
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, 'process.log')

    logger = logging.getLogger('batch_graft')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(console_handler)

    finished_ids, failed_ids = load_finished_ids(log_file)
    skip_ids = finished_ids - failed_ids

    pdb_df = pd.read_csv(pdb_list_file, sep='\t', dtype=str)
    if 'pdb_id' not in pdb_df.columns:
        raise KeyError("Input list must contain a 'pdb_id' column")

    total_count = len(pdb_df)
    logger.info('Loaded %s PDB IDs from %s', total_count, pdb_list_file)

    if skip_ids:
        orig_count = len(pdb_df)
        pdb_df = pdb_df[~pdb_df['pdb_id'].isin(skip_ids)].reset_index(drop=True)
        logger.info(
            'Detected %s finished PDB IDs in existing log. Filtering list: %s -> %s remaining.',
            len(skip_ids),
            orig_count,
            len(pdb_df),
        )

    pending_ids = pdb_df['pdb_id'].tolist()
    if not pending_ids:
        logger.info('No pending PDB IDs to process.')
        return

    if select_mode == 'table':
        tokenized_folder = prepare_table_tokenized_folder(
            cdr_table, tokenized_folder, output_path
        )
        logger.info(
            'Table mode: using %s rows as direct FR recipients; global database search is disabled.',
            len(pd.read_csv(cdr_table, sep='\t', usecols=['PDBChain'])),
        )

    config = BatchConfig(
        pdb_db=pdb_db,
        seqs_file=seqs_file,
        tokenized_folder=tokenized_folder,
        cdr_table=cdr_table,
        output_path=output_path,
        structure_sample=structure_sample,
        structure_batch_size=structure_batch_size,
        top_n=top_n,
        cpu_num=cpu_num,
        workers=max(1, workers),
        select_mode=select_mode,
        s_type=s_type,
        len_limit=len_limit,
        s_temperature=s_temperature,
        distance_limit=distance_limit,
        seed=seed,
        dupl_pdb=dupl_pdb,
        side_chain_max=side_chain_max,
        bf16=bf16,
    )

    gpu_ids = list(gpu_ids) or ['']
    task_buckets = distribute_tasks(pending_ids, gpu_ids)
    logger.info(
        'Starting batch with %s GPUs, %s workers/GPU, total parallel slots=%s.',
        len(gpu_ids),
        config.workers,
        len(gpu_ids) * config.workers,
    )

    start_time = time.time()
    ctx = mp.get_context('spawn')
    result_queue: mp.Queue = ctx.Queue()
    processes = []

    for gpu_id, bucket in zip(gpu_ids, task_buckets):
        if not bucket:
            continue
        process = ctx.Process(
            target=gpu_worker_entry,
            args=(gpu_id, bucket, config, result_queue),
            name=f'gpu-worker-{gpu_id}',
        )
        process.start()
        processes.append(process)

    expected_results = sum(len(bucket) for bucket in task_buckets if bucket)
    completed = 0
    while completed < expected_results:
        result: TaskResult = result_queue.get()
        completed += 1
        if result.success:
            logger.info('Finished processing PDB ID: %s', result.pdb_id)
        else:
            logger.error('Failed processing PDB ID: %s -- %s', result.pdb_id, result.message)

    for process in processes:
        process.join()

    runtime = time.time() - start_time
    logger.info('Total runtime: %.2f seconds', runtime)


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch Graft CDR with persistent ESM3 per GPU')
    parser.add_argument('pdb_list_file', type=str, help='Path to multiple target grafting nanobodies file')
    parser.add_argument('pdb_db', type=str, help='Path to the nanobodies database folder')
    parser.add_argument('seqs_file', type=str, help='Path to the nanobodies database FASTA')
    parser.add_argument('cdr_table', type=str, help='Path to the CDR information file')
    parser.add_argument('output_path', type=str, help='Path to the output directory')
    parser.add_argument('--tokenized_folder', type=str, default=None, help='Pre-tokenized connector database folder')
    parser.add_argument('--gpu', type=str, default='', help='Legacy single GPU id')
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU ids, e.g. 0,1,2')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent task slots per GPU')
    parser.add_argument('--ss', type=int, default=3, help='Number of structure samples per target')
    parser.add_argument('--ss_batch', type=int, default=1, help='GPU mini-batch size for same-prompt structure samples')
    parser.add_argument('--cpu', type=int, default=8, help='Total CPU threads budget per GPU process')
    parser.add_argument('--top', type=int, default=100, help='Top N candidates / side-chain cap input')
    parser.add_argument(
        '--mode', type=str, default='all',
        choices=['table', 'all', 'foldseek', 'mmseqs', 'connector'],
        help='Selection mode; table uses cdr_table rows directly as FR recipients without database search'
    )
    parser.add_argument('--type', type=int, default=1, help='Connector type when mode=connector')
    parser.add_argument('--len_limit', type=int, default=180, help='Maximum prompt length')
    parser.add_argument('--temperature', type=float, default=0.7, help='Structure generation temperature')
    parser.add_argument('--distance_limit', type=float, default=10.0, help='Connector distance filter threshold')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--dupl_pdb', action='store_true', help='Duplicate PDB rows before generation')
    parser.add_argument('--side_chain_max', type=int, default=10, help='Maximum side-chain structures per target')
    parser.add_argument('--bf16', action='store_true', help='Cast ESM3 to bfloat16 for faster GPU generation')

    args = parser.parse_args()
    gpu_ids = parse_gpu_list(args.gpus, args.gpu)

    run_batch(
        pdb_list_file=args.pdb_list_file,
        pdb_db=args.pdb_db,
        seqs_file=args.seqs_file,
        tokenized_folder=args.tokenized_folder,
        cdr_table=args.cdr_table,
        output_path=args.output_path,
        gpu_ids=gpu_ids,
        structure_sample=args.ss,
        structure_batch_size=args.ss_batch,
        top_n=args.top,
        cpu_num=args.cpu,
        workers=args.workers,
        select_mode=args.mode,
        s_type=args.type,
        len_limit=args.len_limit,
        s_temperature=args.temperature,
        distance_limit=args.distance_limit,
        seed=args.seed,
        dupl_pdb=args.dupl_pdb,
        side_chain_max=args.side_chain_max,
        bf16=args.bf16,
    )


if __name__ == '__main__':
    main()
