import argparse
import sys
import shutil
import subprocess
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from mpl_toolkits.axes_grid1 import make_axes_locatable

def parse_args():
    parser = argparse.ArgumentParser(
        description="分析序列Pair-Pair差异，支持MAFFT比对、自定义排序、标签简化及配色调整。"
    )
    
    # 基础输入输出
    parser.add_argument("-i", "--input", required=True, help="输入FASTA文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出文件前缀")
    
    # MAFFT 相关
    parser.add_argument("-m", "--mafft", nargs='?', const="mafft", default=None,
                        help="调用MAFFT进行比对。可指定路径。若不指定则默认尝试系统环境变量中的'mafft'。")
    parser.add_argument("--save-aligned", action="store_true", default=True,
                        help="是否保存Alignment后的FASTA文件 (默认: True)")
    
    # 绘图内容控制
    parser.add_argument("--order", help="指定ID顺序的列表文件路径（每行一个ID），用于控制热图行列顺序。")
    parser.add_argument("--simplify", action="store_true", 
                        help="简化标签：仅使用 '_' 分隔符的第一个片段作为展示名称 (例如: GeneA_Species_01 -> GeneA)。")
    
    # 绘图样式控制
    parser.add_argument("--fontsize", type=int, default=16, help="热图轴标签字体大小 (默认: 16)。")
    parser.add_argument("--annot-size", type=int, default=14, help="热图中数字标注的字体大小 (默认: 14)。")
    parser.add_argument("--color", choices=['blue-green', 'red', 'viridis', 'magma', 'blue'], default='blue-green',
                        help="热图配色方案：blue-green(蓝绿,默认), red(红), viridis(翠绿), magma(岩浆色), blue(蓝)。")

    return parser.parse_args()

def run_mafft(input_file, mafft_path="mafft"):
    """运行MAFFT并返回结果字符串"""
    if not shutil.which(mafft_path):
        print(f"[ERROR] 找不到 MAFFT 程序: '{mafft_path}'")
        sys.exit(1)

    print(f"[INFO] 正在运行 MAFFT ({mafft_path})...")
    cmd = [mafft_path, "--auto", "--quiet", input_file]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] MAFFT 运行出错:\n{e.stderr}")
        sys.exit(1)

def calculate_diff(seq1, seq2):
    """计算差异数"""
    len1, len2 = len(seq1), len(seq2)
    min_len = min(len1, len2)
    diff_count = abs(len1 - len2)
    
    for k in range(min_len):
        if seq1[k] != seq2[k]:
            diff_count += 1
    return diff_count

def get_color_map(color_choice):
    """映射用户友好的颜色名称到 Matplotlib colormap"""
    maps = {
        'red': 'Reds',
        'blue-green': 'GnBu',  # Green to Blue
        'blue': 'Blues',
        'viridis': 'viridis',
        'magma': 'magma'
    }
    return maps.get(color_choice, 'GnBu')

def main():
    args = parse_args()
    
    # --- 1. 获取/比对序列 ---
    records = []
    
    if args.mafft:
        aligned_str = run_mafft(args.input, args.mafft)
        
        # 保存 Alignment 结果
        if args.save_aligned:
            aln_filename = f"{args.output}_aligned.fasta"
            with open(aln_filename, "w") as f:
                f.write(aligned_str)
            print(f"[INFO] 比对后的序列已保存至: {aln_filename}")
            
        fasta_io = io.StringIO(aligned_str)
        records = list(SeqIO.parse(fasta_io, "fasta"))
    else:
        print(f"[INFO] 读取原始文件: {args.input}")
        records = list(SeqIO.parse(args.input, "fasta"))

    if len(records) < 2:
        print("[ERROR] 序列数量不足，无法分析。")
        sys.exit(1)

    # 提取原始ID和序列
    # 使用字典存储以便后续重排序索引
    id_seq_map = {rec.id: str(rec.seq).upper() for rec in records}
    current_ids = list(id_seq_map.keys())
    
    # --- 2. 处理排序 (Order) ---
    if args.order:
        print(f"[INFO] 正在读取顺序文件: {args.order}")
        try:
            with open(args.order, 'r') as f:
                # 读取非空行并去除空白
                ordered_ids = [line.strip() for line in f if line.strip()]
            
            # 过滤：只保留在FASTA中存在的ID，避免报错
            final_ids = [uid for uid in ordered_ids if uid in id_seq_map]
            
            # 检查是否有丢弃或遗漏
            missing_in_fasta = set(ordered_ids) - set(id_seq_map.keys())
            missing_in_order = set(id_seq_map.keys()) - set(ordered_ids)
            
            if missing_in_fasta:
                print(f"[WARNING] 顺序文件中包含FASTA中不存在的ID (已忽略): {len(missing_in_fasta)} 个")
            if missing_in_order:
                print(f"[WARNING] FASTA中包含顺序文件中未指定的ID (将追加到末尾): {len(missing_in_order)} 个")
                final_ids.extend(list(missing_in_order))
            
            current_ids = final_ids
            
        except Exception as e:
            print(f"[ERROR] 读取顺序文件失败: {e}")
            sys.exit(1)

    n = len(current_ids)
    print(f"[INFO] 将分析 {n} 条序列。")

    # --- 3. 计算差异矩阵 ---
    # 确保根据 current_ids 的顺序构建矩阵
    diff_matrix = np.zeros((n, n), dtype=int)
    seqs_ordered = [id_seq_map[uid] for uid in current_ids]

    for i in range(n):
        for j in range(i + 1, n):
            d = calculate_diff(seqs_ordered[i], seqs_ordered[j])
            diff_matrix[i, j] = d
            diff_matrix[j, i] = d
            
    # 创建 DataFrame
    df = pd.DataFrame(diff_matrix, index=current_ids, columns=current_ids)
    
    # --- 4. 标签简化 (Simplify) ---
    if args.simplify:
        print("[INFO] 正在应用标签简化 (取第一个'_'前的内容)...")
        # 创建映射字典: {old_name: new_name}
        rename_map = {}
        for uid in current_ids:
            new_name = uid.split('_')[0]
            rename_map[uid] = new_name
        
        # 检查简化后是否有重复
        if len(set(rename_map.values())) < len(rename_map):
            print("[WARNING] 标签简化导致了名称重复，热图中将出现相同的标签。")
            
        df.rename(index=rename_map, columns=rename_map, inplace=True)

    # 保存矩阵数据
    out_csv = f"{args.output}_matrix.csv"
    df.to_csv(out_csv)
    print(f"[INFO] 差异矩阵已保存至: {out_csv}")

    # --- 5. 绘制图谱 ---
    print(f"[INFO] 正在绘制热图 (配色: {args.color})...")
    
    # 动态画布大小
    fig_size = max(8, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    
    # 自动决定是否显示数字，也可由用户通过逻辑控制，这里保持阈值
    annot = True if n <= 30 else False
    
    cmap_name = get_color_map(args.color)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    
    # 绘图
    sns.heatmap(
        df, 
        ax=ax,
        annot=annot, 
        fmt="d", 
        cmap=cmap_name, 
        annot_kws={
            "size": args.annot_size,
            "fontweight": "bold"
        },
        square=True,
        linewidths=1.5,
        cbar=True,
        cbar_ax=cax,
        cbar_kws={'label': 'Number of AA Differences'})

    # 标题
    # title_str = "Pairwise AA Differences"
    # if args.mafft:
    #     title_str += " (Aligned)"
    # plt.title(title_str, fontsize=args.fontsize + 4)
    
    # 坐标轴刻度
    ax.set_xticklabels(ax.get_xticklabels(),
                    rotation=45 if not args.simplify else 0,
                    ha='right' if not args.simplify else 'center',
                    fontsize=args.fontsize)

    ax.set_yticklabels(ax.get_yticklabels(),
                    rotation=0,
                    fontsize=args.fontsize)

    # colorbar 字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=args.fontsize)
    cbar.set_label('Number of AA Differences', fontsize=args.fontsize)
    cbar.ax.yaxis.labelpad = 8
    
    plt.tight_layout()

    out_img = f"{args.output}_heatmap.png"
    plt.savefig(out_img, dpi=600)
    print(f"[INFO] 图谱已保存至: {out_img} (DPI 600)")
    print("[INFO] 完成。")

if __name__ == "__main__":
    main()