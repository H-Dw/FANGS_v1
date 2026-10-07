#!/usr/bin/env python3
"""
Merge two TSV tables by an inner join on PDB_ID, then retain rows that rank
within a specified upper percentile of both pTM and
grafted_euclidean_all_raw_embeddings. Percentiles are evaluated in descending
order. The filtered table is written to a new TSV file.

Example:
    python merge_and_filter.py --input1 file1.tsv --input2 file2.tsv \
        --output result.tsv --ptm_percent 0.1 --grafted_percent 0.1
        
    python ../scripts/grafting_generation/extract_overfit_sample.py \
        --input1 6lr7B_grafting_connector_all_temp07_s50_250904/temp_generation/passed_info/passed_generation.tsv \
        --input2 6lr7B_grafting_connector_all_temp07_s50_250904/extract_distance/filter_best/grafted_raw_embeddings.tsv \
        --output 6lr7B_grafting_connector_all_temp07_s50_250904/overfit_samples.tsv
"""

import argparse
import pandas as pd
import sys


def extract_pdb_id(value: str) -> str:
    """
    Recover the effective identifier from a raw PDB_ID entry.

    When the value contains an underscore, the substring after the final
    underscore is returned. Otherwise the original value is retained.
    """
    if pd.isna(value):
        return ""
    s = str(value)
    if "_" in s:
        return s.split("_")[-1]
    return s


def load_and_prepare_tsv(file_path: str) -> pd.DataFrame:
    """
    Read a TSV file and standardize its PDB_ID column.

    Returns
    -------
    pandas.DataFrame
        Table whose PDB_ID values have been normalized and whose empty
        identifiers have been removed.
    """
    try:
        df = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        sys.exit(f"错误：无法读取文件 {file_path}\n{e}")

    if "PDB_ID" not in df.columns:
        sys.exit(f"错误：文件 {file_path} 中缺少 'PDB_ID' 列")

    # Standardize each PDB_ID to its effective identifier.
    df["PDB_ID"] = df["PDB_ID"].apply(extract_pdb_id)
    # Remove rows whose standardized PDB_ID is empty.
    df = df[df["PDB_ID"] != ""].copy()
    return df


def filter_top_percent(df: pd.DataFrame, column: str, percent: float) -> pd.Series:
    """
    Return a Boolean mask for rows in the upper percentile of a column.

    Ranks are fractional and descending. A row is retained when its
    fractional rank is at most ``percent``; for example, 0.1 selects the
    top 10% of values.
    """
    if column not in df.columns:
        sys.exit(f"错误：DataFrame 中缺少列 '{column}'")

    # Fractional rank in descending order (rank / n). Larger values approach rank zero.
    rank_pct = df[column].rank(ascending=False, pct=True, na_option="bottom")
    return rank_pct <= percent


def main():
    parser = argparse.ArgumentParser(
        description="合并两个 TSV 文件，并筛选 pTM 和 grafted_euclidean_all_raw_embeddings 两列均在前指定百分比的行。"
    )
    parser.add_argument("--input1", required=True, help="第一个 TSV 文件路径")
    parser.add_argument("--input2", required=True, help="第二个 TSV 文件路径")
    parser.add_argument("--output", required=True, help="输出 TSV 文件路径")
    parser.add_argument(
        "--ptm_percent",
        type=float,
        default=0.01,
        help="pTM 列的筛选百分比（降序前百分比），默认为 0.1（即前 10%%）",
    )
    parser.add_argument(
        "--grafted_percent",
        type=float,
        default=0.01,
        help="grafted_euclidean_all_raw_embeddings 列的筛选百分比（降序前百分比），默认为 0.1",
    )
    args = parser.parse_args()

    # Require both percentiles to lie in (0, 1].
    if not (0 < args.ptm_percent <= 1):
        sys.exit("错误：--ptm_percent 必须在 (0, 1] 范围内")
    if not (0 < args.grafted_percent <= 1):
        sys.exit("错误：--grafted_percent 必须在 (0, 1] 范围内")

    # Load and standardize both input tables.
    print(f"正在读取 {args.input1} ...")
    df1 = load_and_prepare_tsv(args.input1)
    print(f"正在读取 {args.input2} ...")
    df2 = load_and_prepare_tsv(args.input2)

    # Inner join on the standardized PDB_ID.
    print("正在合并两个文件（基于 PDB_ID 内连接）...")
    merged = pd.merge(df1, df2, on="PDB_ID", how="inner")
    if merged.empty:
        print("警告：合并后为空，无输出。")
        # Write an empty table so that the requested output path still exists.
        merged.to_csv(args.output, sep="\t", index=False)
        return

    # Apply the percentile filter to both ranking columns.
    print("正在筛选 pTM 前 {:.0%} ...".format(args.ptm_percent))
    ptm_mask = filter_top_percent(merged, "pTM", args.ptm_percent)
    print("正在筛选 grafted_euclidean_all_raw_embeddings 前 {:.0%} ...".format(args.grafted_percent))
    grafted_mask = filter_top_percent(merged, "grafted_euclidean_all_raw_embeddings", args.grafted_percent)

    final_mask = ptm_mask & grafted_mask
    result = merged[final_mask]

    print(f"合并后总行数：{len(merged)}")
    print(f"pTM 前 {args.ptm_percent:.0%} 行数：{ptm_mask.sum()}")
    print(f"grafted 前 {args.grafted_percent:.0%} 行数：{grafted_mask.sum()}")
    print(f"最终筛选行数：{len(result)}")

    # Write the filtered table.
    result.to_csv(args.output, sep="\t", index=False)
    print(f"结果已保存至 {args.output}")


if __name__ == "__main__":
    main()