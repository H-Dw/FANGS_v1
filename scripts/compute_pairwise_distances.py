#!/usr/bin/env python3
"""
compute_pairwise_distances.py

用法示例：
python compute_pairwise_distances.py \
    --groups_tsv grouped_input.tsv \
    --emb_tsv raw_embeddings.tsv \
    --pdb_dir ./pdbs \
    --usalign_path ./USalign \
    --out_tsv output_pairs.tsv
"""

import argparse
import subprocess
import re
import os
import itertools
import math
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# helper: parse USalign output to get RMSD
# -------------------------
RMSD_PAT = re.compile(r'RMSD\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)

def parse_rmsd_from_output(txt: str) -> Optional[float]:
    if not txt:
        return None
    m = RMSD_PAT.search(txt)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    return None

# -------------------------
# resolve pdb file path
# -------------------------
def find_pdb_file(pdbchain: str, pdb_dir: str) -> Optional[str]:
    pdir = Path(pdb_dir)

    candidates = []
    # raw
    pc = Path(pdbchain)
    if pc.suffix.lower() == '.pdb':
        candidates.append(pdir / pdbchain)
    else:
        candidates.append(pdir / (pdbchain + '.pdb'))
        candidates.append(pdir / pdbchain)  # in case given full filename without suffix

    for c in candidates:
        if c.exists():
            return str(c.resolve())

    # not found
    return None

# -------------------------
# call USalign to compute RMSD
# -------------------------
def compute_rmsd_using_usalign(usalign_path: str, pdb1: str, pdb2: str, timeout: int = 60) -> Optional[float]:
    cmd = [usalign_path, pdb1, pdb2]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print(f"[USalign] 调用失败: {e}  cmd={' '.join(cmd)}")
        return None

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    rmsd = parse_rmsd_from_output(out)
    if rmsd is None:
        print(f"[USalign] could not find RMSD, return={proc.returncode}, stdout/stderr:\n{out}")
    return rmsd

# -------------------------
# load embeddings table
# -------------------------
def load_embeddings(emb_tsv: str, id_col: str = "ID") -> Dict[str, np.ndarray]:
    """
    载入 raw_embeddings.tsv，返回 id -> np.array(embedding)
    支持两种常见格式：
      A) 多列：ID | emb0 | emb1 | emb2 ...
      B) 一列：ID | embedding  （embedding 为 "1.0,2.0,3.0" 或 "[1.0, 2.0, 3.0]"）
    """
    df = pd.read_csv(emb_tsv, sep='\t', dtype=str)
    if id_col not in df.columns:
        raise ValueError(f"Embeddings table 没有找到 ID 列: {id_col}")

    # 尝试识别 numeric 列（除 ID 以外）
    other_cols = [c for c in df.columns if c != id_col]
    embeddings: Dict[str, np.ndarray] = {}

    # case A: 多 numeric 列
    # 尝试将 other_cols 转为 numeric；如果大多数能转为 numeric，我们就采用这种方式
    if len(other_cols) >= 2:
        numeric_df = df[other_cols].apply(pd.to_numeric, errors='coerce')
        non_na_ratio = numeric_df.notna().sum().sum() / (numeric_df.shape[0] * numeric_df.shape[1])
        if non_na_ratio > 0.5:
            # 采用多列方式
            for _, row in pd.concat([df[[id_col]], numeric_df], axis=1).iterrows():
                key = str(row[id_col]).strip()
                vec = row[other_cols].values.astype(float)
                embeddings[key] = np.asarray(vec, dtype=float)
            return embeddings

    # case B: single column embedding (文本)
    # 找第一个非 ID 列当作 embedding 列（或名为 'embedding'）
    emb_col = None
    if 'embedding' in df.columns:
        emb_col = 'embedding'
    elif len(other_cols) >= 1:
        emb_col = other_cols[0]

    if emb_col is None:
        raise ValueError("无法识别 embeddings 列格式：既没有数值列，也没有 'embedding' 列。")

    for _, row in df.iterrows():
        key = str(row[id_col]).strip()
        raw = row[emb_col]
        if pd.isna(raw):
            continue
        s = str(raw).strip()
        # 去掉中括号
        s = s.strip('[]() ')
        parts = [p.strip() for p in re.split(r'[,\s]+', s) if p.strip() != ""]
        try:
            vec = np.array([float(x) for x in parts], dtype=float)
            embeddings[key] = vec
        except Exception as e:
            # 解析失败跳过该行
            print(f"[Embeddings] 解析失败 id={key} raw='{raw}': {e}")
            continue

    return embeddings

# -------------------------
# compute euclidean distance
# -------------------------
def euclidean(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        print(f"[Distance] Missmatch shape: {a.shape} vs {b.shape}")
        return None
    return float(np.linalg.norm(a - b))

# -------------------------
# main processing
# -------------------------
def process_groups(groups_tsv: str,
                   emb_tsv: str,
                   output_tsv: str,
                   pdb_dir: str = ".",
                   usalign_path: str = "./USalign",
                   member_col: str = "Member",
                   id_col_in_emb: str = "ID",
                   connector_len: int = 12,
                   ):

    gdf = pd.read_csv(groups_tsv, sep='\t', dtype=str).fillna('')
    if member_col not in gdf.columns:
        raise ValueError(f"Groups TSV missed column: {member_col}")

    embeddings = load_embeddings(emb_tsv, id_col=id_col_in_emb)

    all_groups: List[List[str]] = []
    for _, row in gdf.iterrows():
        member_field = str(row.get(member_col, "")).strip()
        if member_field == "":
            continue
        members = [m.strip() for m in member_field.split(',') if m.strip() != ""]

        seen = set()
        members_unique = []
        for m in members:
            if m not in seen:
                seen.add(m)
                members_unique.append(m)
        if len(members_unique) >= 2:
            all_groups.append(members_unique)

    print(f"[Info] From {groups_tsv} extracted {len(all_groups)} group (members >= 2)")

    out_rows = []
    usalign_cache: Dict[Tuple[str,str], Optional[float]] = {}

    for grp_idx, members in enumerate(all_groups, start=1):
        for a, b in itertools.combinations(members, 2):
            p1 = a
            p2 = b

            # resolve pdb files
            pdb1 = find_pdb_file(p1, pdb_dir)
            pdb2 = find_pdb_file(p2, pdb_dir)
            if pdb1 is None or pdb2 is None:
                msg = f"[Warn] PDB missed: {p1}->{pdb1}, {p2}->{pdb2}"
                print(msg + "  filled with None in RMSD")

            # compute RMSD (cache symmetric)
            key = tuple(sorted([p1, p2]))
            if key in usalign_cache:
                rmsd_val = usalign_cache[key]
            else:
                if pdb1 is not None and pdb2 is not None:
                    rmsd_val = compute_rmsd_using_usalign(usalign_path, pdb1, pdb2)
                else:
                    rmsd_val = None
                usalign_cache[key] = rmsd_val

            # embeddings distance: try find by ID key directly (use the entire PDBChain as ID)
            emb1 = embeddings.get(p1)
            emb2 = embeddings.get(p2)
            conn_dist = None
            if emb1 is None or emb2 is None:
                print(f"[Warn] embedding not found for {p1} or {p2}, filled with NaN")
                conn_dist = None
            else:
                conn_dist = euclidean(emb1, emb2)
                conn_dist = conn_dist / math.sqrt(connector_len)

            out_rows.append({
                "PDB1": p1,
                "PDB2": p2,
                "RMSD": "" if rmsd_val is None else rmsd_val,
                "ConnectorDistance": "" if conn_dist is None else conn_dist
            })

    out_df = pd.DataFrame(out_rows, columns=["PDB1", "PDB2", "RMSD", "ConnectorDistance"])
    out_df.to_csv(output_tsv, sep='\t', index=False)
    print(f"[Done] Writed {len(out_df)} rows into {output_tsv}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Calculation of pairwise RMSD and connector embedding distance")
    parser.add_argument("--groups", required=True, help="TSV file including Member column")
    parser.add_argument("--emb", required=True, help="raw_embeddings.tsv, including ID and embeddings column")
    parser.add_argument("--out", default="output_pairs.tsv", help="Output")
    parser.add_argument("--pdb_dir", default=".", help="PDB folder")
    parser.add_argument("--usalign_path", default="./USalign", help="USalign path")
    parser.add_argument("--member_col", default="Member", help="Member column name")
    parser.add_argument("--id_col_in_emb", default="ID", help="ID column in emb")
    parser.add_argument("--connector", type=int, default=12, help="Connector length default 12")
    args = parser.parse_args()

    process_groups(
        groups_tsv=args.groups,
        emb_tsv=args.emb,
        output_tsv=args.out,
        pdb_dir=args.pdb_dir,
        usalign_path=args.usalign_path,
        member_col=args.member_col,
        id_col_in_emb=args.id_col_in_emb,
        connector_len=args.connector,
    )

if __name__ == "__main__":
    main()
