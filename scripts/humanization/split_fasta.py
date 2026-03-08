#!/usr/bin/env python3
"""
split_fasta.py

通过命令行读取一个 FASTA 文件，将其中的每条序列分别保存为单独的文件，输出到指定目录。

Usage:
    python split_fasta.py -i input.fasta -o output_dir
"""
import os
import argparse
from Bio import SeqIO

def parse_args():
    parser = argparse.ArgumentParser(
        description="将 FASTA 文件中的每条序列分割并保存为单独文件"
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help="输入的 FASTA 文件路径"
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help="输出目录，若不存在则创建"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    output_dir = args.output

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 解析并写入每条序列
    for record in SeqIO.parse(input_path, "fasta"):
        # 使用序列ID作为文件名，保留FASTA扩展名
        filename = f"{record.id}.fasta"
        out_path = os.path.join(output_dir, filename)
        record.id = "A|protein|"
        record.description = ''

        # 写入单个序列
        with open(out_path, "w") as handle:
            SeqIO.write(record, handle, "fasta")
        # print(f"写入: {out_path}")

if __name__ == '__main__':
    main()
