import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def process_data(file_path):
    # 读取TSV文件
    data = pd.read_csv(file_path, sep='\t')

    # 按pTM降序排序
    data = data.sort_values(by='pTM', ascending=False)

    # 根据PDB_ID去重，保留每个PDB_ID对应的最大pTM值的行
    data = data.drop_duplicates(subset='PDB_ID', keep='first')

    return data
def calculate_statistics(data, output_dir):
    # 计算cRMSD的统计值
    cdr_columns = ['CDR1_cRMSD', 'CDR2_cRMSD', 'CDR3_cRMSD']
    stats = {}
    for col in cdr_columns:
        stats[col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'count': data[col].count()  # 添加数量统计
        }
    
    # 计算pTM的统计值
    stats['pTM'] = {
        'mean': data['pTM'].mean(),
        'std': data['pTM'].std(),
        'count': data['pTM'].count()  # 添加数量统计
    }

    print("\nStatistics:")
    print("\ncRMSD Statistics:")
    for cdr in ['CDR1_cRMSD', 'CDR2_cRMSD', 'CDR3_cRMSD']:
        print(f"{cdr}:")
        print(f"  Count: {stats[cdr]['count']}")  # 打印数量
        print(f"  Mean:  {stats[cdr]['mean']:.4f}")
        print(f"  Std:   {stats[cdr]['std']:.4f}")
    
    print("\npTM Statistics:")
    print(f"  Count: {stats['pTM']['count']}")  # 打印数量
    print(f"  Mean:  {stats['pTM']['mean']:.4f}")
    print(f"  Std:   {stats['pTM']['std']:.4f}")

    stats_file = os.path.join(output_dir, 'statistics.txt')
    
    with open(stats_file, 'w') as f:
        f.write("Statistics:\n\n")
        f.write("cRMSD Statistics:\n")
        for cdr in ['CDR1_cRMSD', 'CDR2_cRMSD', 'CDR3_cRMSD']:
            f.write(f"{cdr}:\n")
            f.write(f"  Count: {stats[cdr]['count']}\n")
            f.write(f"  Mean:  {stats[cdr]['mean']:.4f}\n")
            f.write(f"  Std:   {stats[cdr]['std']:.4f}\n\n")
        
        f.write("pTM Statistics:\n")
        f.write(f"  Count: {stats['pTM']['count']}\n")
        f.write(f"  Mean:  {stats['pTM']['mean']:.4f}\n")
        f.write(f"  Std:   {stats['pTM']['std']:.4f}\n")
    
    print(f"Statistics saved to: {stats_file}")
    
    return stats

def plot_boxplot_with_errorbar(data, output_dir):
    # 提取需要的列并转换为长格式
    cdr_columns = ['CDR1_cRMSD', 'CDR2_cRMSD', 'CDR3_cRMSD']
    long_data = data[cdr_columns].melt(var_name='CDR Region', value_name='cRMSD Value')

    # 修改横坐标名称
    long_data['CDR Region'] = long_data['CDR Region'].replace({
        'CDR1_cRMSD': 'CDR1',
        'CDR2_cRMSD': 'CDR2',
        'CDR3_cRMSD': 'CDR3'
    })

    # # 计算均值和标准差
    # means = long_data.groupby('CDR Region')['cRMSD Value'].mean()
    # stds = long_data.groupby('CDR Region')['cRMSD Value'].std()

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=long_data, x='CDR Region', y='cRMSD Value', palette=['#7AB656', '#7E99F4', '#CC7C71'])

    # # 添加误差线
    # for i, (mean, std) in enumerate(zip(means, stds)):
    #     plt.errorbar(i, mean, yerr=std, fmt='o', color='black', capsize=5, label=f"Mean ± Std ({means.index[i]})")

    # 设置图表标题和标签
    # plt.xlabel('CDR Regions', fontsize=14, fontweight='bold')
    plt.ylabel('cRMSD', fontsize=14, fontweight='bold')
    # plt.title('Boxplot with Error Bars of cRMSD Values for CDR Regions')

    # 保存箱线图为PNG文件
    output_plot_path = os.path.join(output_dir, 'cdr_rmsd_boxplot_with_errorbar.png')
    plt.savefig(output_plot_path, dpi=600)
    print(f"Boxplot with error bars saved to: {output_plot_path}")

def save_processed_data(data, output_dir):
    # 保存处理后的数据为CSV文件
    output_data_path = os.path.join(output_dir, 'processed_data.csv')
    data.to_csv(output_data_path, index=False, sep='\t')
    print(f"Processed data saved to: {output_data_path}")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Process TSV file and plot violin chart and boxplot.")
    parser.add_argument('input_file', type=str, help="Path to the input TSV file")
    parser.add_argument('output_dir', type=str, help="Directory to save the output files")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 处理数据
    processed_data = process_data(args.input_file)

    stat = calculate_statistics(processed_data, args.output_dir)

    # # 打印处理后的数据（可选）
    # print("Processed Data:")
    # print(processed_data)

    # 保存处理后的数据
    save_processed_data(processed_data, args.output_dir)

    plot_boxplot_with_errorbar(processed_data, args.output_dir)

if __name__ == "__main__":
    main()