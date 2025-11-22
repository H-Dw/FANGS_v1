#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 Fasta 文件中蛋白序列的 Eisenberg 全局疏水性、疏水矩和 pH 下净电荷，并将结果输出到指定的 TSV 文件。
"""
import os
import argparse
import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv

# Eisenberg 等人给出的氨基酸疏水性标度（Eisenberg et al., 1984）
EISENBERG_SCALE = {
    'A':  0.62, 'C':  0.29, 'D': -0.90, 'E': -0.74,
    'F':  1.19, 'G':  0.48, 'H': -0.40, 'I':  1.38,
    'K': -1.50, 'L':  1.06, 'M':  0.64, 'N': -0.78,
    'P':  0.12, 'Q': -0.85, 'R': -2.53, 'S': -0.18,
    'T': -0.05, 'V':  1.08, 'W':  0.81, 'Y':  0.26
}

def global_hydrophobicity(seq: str) -> float:
    """平均 Eisenberg 疏水性"""
    vals = [EISENBERG_SCALE.get(aa, 0.0) for aa in seq]
    return sum(vals) / len(vals) if seq else 0.0

def calc_pI(seq: str) -> float:
    pa = ProteinAnalysis(seq)
    return pa.isoelectric_point()

def global_hydrophobic_moment(seq: str, angle: float = 100.0) -> float:
    """
    全局疏水矩：
      μ = (1/N) * sqrt( (Σ Hi cos(iθ))^2 + (Σ Hi sin(iθ))^2 )
    其中 θ 为每残基转过的角度（α-螺旋约 100°；β-折叠约 180°）
    """
    if not seq:
        return 0.0
    rad = np.deg2rad(angle)
    xs, ys = 0.0, 0.0
    for i, aa in enumerate(seq):
        Hi = EISENBERG_SCALE.get(aa, 0.0)
        xs += Hi * np.cos(i * rad)
        ys += Hi * np.sin(i * rad)
    return np.sqrt(xs**2 + ys**2) / len(seq)


def global_charge(seq: str, pH: float = 7.4) -> float:
    """使用 ProtParam 计算在给定 pH 下的净电荷"""
    pa = ProteinAnalysis(seq)
    return pa.charge_at_pH(pH)


def analyze_fasta(fasta_path: Path, out_tsv: Path, pH: float):
    """
    解析 fasta 文件，计算每条序列的指标，并写入 TSV 文件
    """
    # 创建输出目录
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    with out_tsv.open('w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        # 写入表头
        header = ['ID', 'Length', 'Hydrophobicity', 'HydrophobicMoment', f'Charge@pH{pH}', 'PI']
        writer.writerow(header)

        # 处理每条序列
        for rec in SeqIO.parse(str(fasta_path), 'fasta'):
            seq = str(rec.seq)
            length = len(seq)
            hydro = global_hydrophobicity(seq)
            moment = global_hydrophobic_moment(seq, angle=180.0)
            charge = global_charge(seq, pH)
            pi = calc_pI(seq)
            writer.writerow([rec.id, length, f'{hydro:.4f}', f'{moment:.4f}', f'{charge:.4f}', f'{pi: .4f}'])

    print(f"Results written to: {out_tsv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="计算 Fasta 中蛋白序列的全局疏水性/疏水矩和 pH 下净电荷，并输出 TSV 文件"
    )
    parser.add_argument('fasta', type=Path, help='输入 Fasta 文件路径')
    parser.add_argument('-p', '--pH', type=float, default=7.4,
                        help='净电荷计算 pH 值，默认 7.4')
    parser.add_argument('-o', '--out', type=Path, required=True,
                        help='输出 TSV 文件路径')
    args = parser.parse_args()
    analyze_fasta(args.fasta, args.out, args.pH)
