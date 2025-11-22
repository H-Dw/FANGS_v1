
import pandas as pd
import os, shutil, argparse, time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import graft_CDR_connector_batchversion as graft_CDR

import os
import time
import logging
import re
import pandas as pd


def build_cdr_file(pdb_id, cdr_table, output_path):
    cdr_df = pd.read_csv(cdr_table, sep='\t')

    target_row = cdr_df[cdr_df['PDBChain'] == pdb_id]
    if target_row.empty:
        raise ValueError(f"No entry for PDBChain={pdb_id} in {cdr_table}")

    output_info = target_row[['Sequence', 'CDR1', 'CDR2', 'CDR3', 'PDBChain']]

    os.makedirs(output_path, exist_ok=True)
    output_filename = os.path.join(output_path, f'{pdb_id}.tsv')
    output_info.to_csv(
        output_filename,
        sep='\t',
        index=False,
        header=['Sequence', 'CDR1', 'CDR2', 'CDR3', 'PDBChain']
    )

    return output_filename

def process_pdb(pdb_id, pdb_db, seqs_file, tokenized_folder, cdr_table, output_path, gpu, structure_sample, top_n, cpu_num):
    try:
        # Build folder
        pdb_folder = os.path.join(output_path, pdb_id)
        if not os.path.exists(pdb_folder):
            os.makedirs(pdb_folder)

        # Extract Nb_pdb_path
        ori_pdb_filename = os.path.join(pdb_db, f'{pdb_id}.pdb')
        pdb_filename = os.path.join(pdb_folder, f'{pdb_id}.pdb')
        shutil.copy(ori_pdb_filename, pdb_filename)

        # Build Nb_CDR_file
        cdr_filename = build_cdr_file(pdb_id=pdb_id, cdr_table=cdr_table, output_path=pdb_folder)

        # Find candidate grafting FR
        graf_log_file = os.path.join(pdb_folder, 'graft_process.log')
        with open(graf_log_file, "w") as file:
            with redirect_stdout(file):
                graft_CDR.main(
                    Nb_pdb_path=pdb_filename,
                    Nb_CDR_file=cdr_filename,
                    FR_pdb_folder=pdb_db,
                    FR_CDR_file=cdr_table,
                    output_path=pdb_folder,
                    ag_pdb_path='', 
                    structure_sample=structure_sample, 
                    gpu=gpu, 
                    cpu_num=cpu_num, 
                    non_generation=False, 
                    select_mode='all', 
                    type=1,
                    top_n=top_n,
                    len_limit=180,
                    seqs_file=seqs_file,
                    tokenized_folder=tokenized_folder
                )
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")

def main(pdb_list_file, pdb_db, seqs_file, tokenized_folder, cdr_table, output_path, gpu, structure_sample, top_n, cpu_num):
    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 日志文件路径
    log_file = os.path.join(output_path, "process.log")
    
    # 从已有日志中提取已完成的 PDB ID
    finished_ids = set()
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                # 匹配形如 "Finished processing PDB ID: 1ABC"
                m = re.search(r"Finished processing PDB ID:\s*([0-9A-Za-z]{5})", line)
                if m:
                    finished_ids.add(m.group(1))
    
    # 配置日志，将 INFO 及以上级别写入文件
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 如果 logger 已经有 handler，先清除，避免重复记录
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    start_time = time.time()
    
    # 读取 PDB 列表
    pdb_df = pd.read_csv(pdb_list_file, sep="\t", dtype=str)
    total_count = len(pdb_df)
    logger.info(f"Loaded {total_count} PDB IDs from {pdb_list_file}")
    
    # 如果已有已完成的 ID，则去除
    if finished_ids:
        orig_count = len(pdb_df)
        pdb_df = pdb_df[~pdb_df['pdb_id'].isin(finished_ids)].reset_index(drop=True)
        filtered_count = len(pdb_df)
        logger.info(f"Detected {len(finished_ids)} finished PDB IDs in existing log. "
                    f"Filtering list: {orig_count} → {filtered_count} remaining.")
        logger.info(f"Previous procession"
                    f"Finished processing PDB ID: {finished_ids}")
    
    # 逐个处理剩余的 PDB ID
    for _, row in pdb_df.iterrows():
        pdb_id = row['pdb_id']
        try:
            process_pdb(pdb_id, pdb_db, seqs_file, tokenized_folder, cdr_table, output_path, gpu, structure_sample, top_n, cpu_num)
            logger.info(f"Finished processing PDB ID: {pdb_id}")
        except Exception as e:
            logger.error(f"Error processing PDB ID {pdb_id}: {e}", exc_info=True)
    
    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"Total runtime: {runtime:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Graft CDR")
    parser.add_argument("pdb_list_file", type=str, help="Path to mutiple target grafting nanobodies file path")
    parser.add_argument("pdb_db", type=str, help="Path to the nanobodies database path")
    parser.add_argument("seqs_file", type=str, help="Path to the nanobodies database path")
    parser.add_argument("cdr_table", type=str, help="Path to the CDR information file path")
    parser.add_argument("output_path", type=str, help="Path to the output directory")
    parser.add_argument("--tokenized_folder", type=str, default=None, help="[Only work when select connector as mode] tokenized files folder for database")
    parser.add_argument("--gpu", type=str, default='', help="gpu_id")
    parser.add_argument("--ss", type=int, default=3, help="Number for structure_sample")
    parser.add_argument("--cpu", type=int, default=8, help="cpu_num")
    parser.add_argument("--top", type=int, default=100, help="top_n selected in each feature")


    args = parser.parse_args()

    main(args.pdb_list_file, args.pdb_db, args.seqs_file, args.tokenized_folder, args.cdr_table, args.output_path, args.gpu, args.ss, args.top, args.cpu)