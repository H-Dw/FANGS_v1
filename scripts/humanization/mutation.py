#!/usr/bin/env python3
import pandas as pd
import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Apply Kabat-position mutations to numbering table and output FASTA")
    p.add_argument("-t", "--target", required=True, help="Target CSV with numbering (first column 'Position')")
    p.add_argument("-m", "--mutation", required=True, help="Mutation CSV with columns 'Position','MutAA'")
    p.add_argument("-o", "--output", required=True, help="Output FASTA file")
    return p.parse_args()

def main():
    args = parse_args()
    # 读取 target 编号表
    try:
        df = pd.read_csv(args.target, dtype=str)
    except Exception as e:
        sys.exit(f"Error reading target CSV: {e}")

    if 'Position' not in df.columns:
        sys.exit("Error: target CSV must have a 'Position' column")

    # 读取 mutation 表
    try:
        mut_df = pd.read_csv(args.mutation, dtype=str)
    except Exception as e:
        sys.exit(f"Error reading mutation CSV: {e}")

    if not {'Position','MutAA'}.issubset(mut_df.columns):
        sys.exit("Error: mutation CSV must have 'Position' and 'MutAA' columns")

    # 对每个突变点，批量替换所有序列列
    seq_cols = [c for c in df.columns if c != 'Position']
    df_mod = df.copy()
    for _, row in mut_df.iterrows():
        pos = row['Position']
        aa  = row['MutAA']
        mask = df_mod['Position'] == pos
        if not mask.any():
            sys.stderr.write(f"Warning: position {pos} not found in target table, skipping\n")
            continue
        # 替换这一行所有序列列为突变氨基酸
        df_mod.loc[mask, seq_cols] = aa

    # 输出 FASTA：每一列拼成序列，去除 '-'，ID 是列名
    with open(args.output, 'w') as out_f:
        for col in seq_cols:
            seq = ''.join(df_mod[col].tolist()).replace('-', '')
            out_f.write(f">{col}\n")
            # 按 70 字符换行
            for i in range(0, len(seq), 70):
                out_f.write(seq[i:i+70] + "\n")

    print(f"Written mutated sequences to {args.output}")

if __name__ == "__main__":
    main()
