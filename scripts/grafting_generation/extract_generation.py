import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


def residuals(y, X):
    """对 y ~ X 做线性回归，返回 y 的残差"""
    model = LinearRegression().fit(X, y)
    return y - model.predict(X)

def plot_generations(connector_df, output_path):
    # 准备数据
    x = connector_df['cRMSD']
    y = connector_df['pTM']

    # 计算 Spearman r、p-value 和 R²
    r, p_val = spearmanr(x, y)
    r2 = r**2

    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(8, 6))

    # 散点
    sns.scatterplot(x=x, y=y, s=50, alpha=0.7, ax=ax)
    # 回归线
    sns.regplot(x=x, y=y, scatter=False, ci=95,
                line_kws={'color': 'red', 'lw': 1.5}, ax=ax)
    # 虚线
    ax.axhline(0.8, color='blue', linestyle='--', lw=1, label='pTM = 0.8')
    ax.axvline(1.5, color='green', linestyle='--', lw=1, label='cRMSD = 1.5')

    title_str = f"$r={r:.3f},\ \;R^2={r2:.3f},\ \;p={p_val:.3g}$"
    ax.set_title(title_str, loc='center', fontsize=12, pad=10, fontstyle='italic')
    ax.set_xlabel('cRMSD', fontsize=14)
    ax.set_ylabel('pTM', fontsize=14)
    ax.legend(loc='lower left', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    output_file = os.path.join(output_path, "generations_distribution.tif")
    plt.savefig(output_file, format='tiff', dpi=600)
    plt.close(fig)

def read_generations(pdb_id, pdb_folder, region_list, embe_list):

    generation_tsv = os.path.join(
        pdb_folder, 'temp_generation', 'all_info', 'all_generation.tsv'
    )
    generation_df = pd.read_csv(generation_tsv, sep='\t')
    for region in region_list:
        for embe in embe_list:
            orig_path = os.path.join(
                pdb_folder, 'similarity', f'{embe}_similarity_{region}.tsv'
            )
            gen_path = os.path.join(
                pdb_folder, 'generation_tokenized_structure',
                f'{embe}_similarity_{region}.tsv'
            )
            orig_df = pd.read_csv(orig_path, sep='\t')
            gen_df = pd.read_csv(gen_path, sep='\t')
            # rename columns
            orig_df = orig_df.rename(columns={
                'euclidean': f'original_euclidean_{region}_{embe}',
                'cosine':    f'original_cosine_{region}_{embe}'
            })
            gen_df = gen_df.rename(columns={
                'euclidean': f'grafted_euclidean_{region}_{embe}',
                'cosine':    f'grafted_cosine_{region}_{embe}'
            })
            generation_df = generation_df.merge(
                orig_df, how='inner', left_on='PDB_ID', right_on='pdb_id'
            ).drop(columns=['pdb_id'])
            generation_df['pdb_id_stripped'] = (
                generation_df['Filename'].str.replace(r'\.pdb$', '', regex=True)
            )
            generation_df = generation_df.merge(
                gen_df, how='inner', left_on='pdb_id_stripped', right_on='pdb_id'
            ).drop(columns=['pdb_id_stripped', 'pdb_id'])
    generation_df['PDB_ID'] = generation_df['PDB_ID'].apply(
        lambda x: f"{pdb_id}_{x}"
    )

    return generation_df

def norm_distance(df, stage_list, region_list, embe_list):
    for stage in stage_list:
        for region in region_list:
            for embe in embe_list:
                length = 12 if region == 'all' else 4
                col = f'{stage}_euclidean_{region}_{embe}'
                df[col] = df[col] / np.sqrt(length)
    for region in region_list:
        for embe in embe_list:
            df[f'change_euclidean_{region}_{embe}'] = (
                df[f'original_euclidean_{region}_{embe}'] -
                df[f'grafted_euclidean_{region}_{embe}']
            ).abs()
    return df

def split_dataframe(
    df: pd.DataFrame,
    prefixes: list[str] = None,
    suffixes: list[str] = None,
    id_column: str = 'PDB_ID',
    file_column: str | None = None
) -> dict[str, pd.DataFrame]:
    """
    拆分 df，每个子表至少保留 id_column，
    如果 file_column 被指定且存在，则也保留它。
    """
    splits = {}
    base_cols = [id_column]
    if file_column and file_column in df.columns:
        base_cols.append(file_column)

    if prefixes:
        for pref in prefixes:
            cols = [c for c in df.columns if c.startswith(pref)]
            if cols:
                splits[pref] = df[base_cols + cols].copy()
    elif suffixes:
        for suf in suffixes:
            cols = [c for c in df.columns if c.endswith(suf)]
            if cols:
                splits[suf] = df[base_cols + cols].copy()
    else:
        raise ValueError("Need prefixes or suffixes")
    return splits

def split_output_df(
    embe_df: pd.DataFrame,
    embe_type: str,
    prefixes: list[str],
    output_path: str,
    file_column: str | None = 'Filename'
):
    """
    和原来几乎一样，只要 file_column=None 就不会去找 Filename。
    """
    stage_dict = split_dataframe(
        embe_df,
        prefixes=prefixes,
        id_column='PDB_ID',
        file_column=file_column
    )
    for stage_type, stage_df in stage_dict.items():
        # 排序并去重
        col_name = f"{stage_type}_euclidean_all_{embe_type}"
        if col_name in stage_df.columns:
            stage_df = stage_df.sort_values(by=col_name, ascending=True)
        else:
            print(f'[ERROR] Target filter column do not exist: {col_name}')
        out_file = os.path.join(output_path, f"{stage_type}_{embe_type}.tsv")
        stage_df.to_csv(out_file, sep='\t', index=False)

def extract_distance(target_list_path, generation_folder, stage_list=None, region_list=None, embe_list=None, output_path=None, limitation=10):
    df_targets = pd.read_csv(target_list_path)
    stage_list = ['original', 'grafted'] if stage_list is None else stage_list
    region_list = ['all', 'connector1', 'connector2', 'connector3'] if region_list is None else region_list
    embe_list = ['raw_embeddings', 'pre_q_embeddings', 'ca_distance'] if embe_list is None else embe_list

    # Extracted and merged generations file
    all_results = []
    success = fail = 0
    for pdb_id in df_targets['pdb_id']:
        # Identified batch or single generation
        pdb_folder = (generation_folder if len(df_targets)==1 else os.path.join(generation_folder, pdb_id))

        try:
            generation_df = read_generations(pdb_id, pdb_folder, region_list, embe_list)
            all_results.append(generation_df)
            success += 1
        except Exception as e:
            print(f"[ERROR] extract {pdb_id} fail: {e}")
            fail += 1

    if not all_results:
        print("[ERROR] No valid generation data found. Exit.")
        sys.exit(1)

    df_gen = pd.concat(all_results, ignore_index=True)
    print(f"Passed files: {success}\t Failed files: {fail}")

    # Plot generation distribution
    os.makedirs(output_path, exist_ok=True)
    plot_generations(df_gen, output_path)

    # Length-normalized for distance
    df_norm = norm_distance(df_gen, stage_list, region_list, embe_list)

    # Stored based on embeddings type
    embe_dict = split_dataframe(df_norm, suffixes=embe_list, file_column='Filename')
    for embe_type, embe_df in embe_dict.items():
        full_output = os.path.join(output_path, 'full')
        filt_output = os.path.join(output_path, 'filter')
        os.makedirs(full_output, exist_ok=True)
        os.makedirs(filt_output, exist_ok=True)

        full_avge_output = os.path.join(output_path, 'full_avge')
        filt_avge_output = os.path.join(output_path, 'filter_avge')
        os.makedirs(full_avge_output, exist_ok=True)
        os.makedirs(filt_avge_output, exist_ok=True)

        full_best_output = os.path.join(output_path, 'full_best')
        filt_best_output = os.path.join(output_path, 'filter_best')
        os.makedirs(full_best_output, exist_ok=True)
        os.makedirs(filt_best_output, exist_ok=True)

        print(f'embe_type: {embe_type}')
        print(f'column name: {embe_df.columns}')
        split_output_df(embe_df, embe_type, prefixes=['change', 'original', 'grafted'], output_path=full_output, file_column='Filename')

        # Averaged
        cols_to_avge = [c for c in embe_df.columns if c != 'Filename' and c != 'PDB_ID']
        embe_mean_df = (
            embe_df
            .groupby('PDB_ID', as_index=False)[cols_to_avge]
            .mean()
        )
        split_output_df(embe_mean_df, embe_type, prefixes=['change', 'original', 'grafted'], output_path=full_avge_output, file_column=None) # Filename is meansless after averaging

        filt_df = embe_df[(embe_df[f'change_euclidean_all_{embe_type}'] <= limitation)].copy()

        filt_mean_df = (
            filt_df
            .groupby('PDB_ID', as_index=False)[cols_to_avge]
            .mean()
        )
        split_output_df(filt_mean_df, embe_type, prefixes=['change', 'original', 'grafted'], output_path=filt_avge_output, file_column=None)

        # Best - Updated at 250831
        embe_best_df = (
            embe_df
            .groupby('PDB_ID', as_index=False)[cols_to_avge]
            .min() #  对指定列计算最小值
        )
        split_output_df(embe_best_df, embe_type, prefixes=['change', 'original', 'grafted'], output_path=full_best_output, file_column=None)

        filt_best_df = (
            filt_df
            .groupby('PDB_ID', as_index=False)[cols_to_avge]
            .min()
        )
        split_output_df(filt_best_df, embe_type, prefixes=['change', 'original', 'grafted'], output_path=filt_best_output, file_column=None)

        # # Duplication
        # col_name = f"grafted_euclidean_all_{embe_type}"
        # stage_df = filt_df.sort_values(by=col_name, ascending=True)
        # dupl_df = stage_df.drop_duplicates(subset=['PDB_ID'], keep='first')
        # split_output_df(dupl_df, embe_type, prefixes=['change', 'original', 'grafted'], output_path=filt_output, file_column='Filename')

        split_output_df(filt_df, embe_type, prefixes=['change', 'original', 'grafted'], output_path=filt_output, file_column='Filename')


def main():
    parser = argparse.ArgumentParser(description="Extract and analyze distance data by PDB IDs")
    parser.add_argument('target_list', help="Path to TSV file listing target pdb_id column")
    parser.add_argument('generation_folder', help="Base folder containing generation subfolders per PDB ID")
    parser.add_argument('output_path', help="Folder to save plots and split TSVs")
    parser.add_argument('--limitation', type=float, default=10.0, help="Limitation of connector change (recommand: 10)")
    args = parser.parse_args()

    extract_distance(
        target_list_path=args.target_list,
        generation_folder=args.generation_folder,
        output_path=args.output_path,
        limitation=args.limitation
    )

if __name__ == '__main__':
    main()

