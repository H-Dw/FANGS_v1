import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import os

def split_connector(target_df):
    first_column = target_df.iloc[:, 0]  # 首列-id
    data_to_scale = target_df.iloc[:, 1:]  # 其余列-embedding
    
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(data_to_scale)
    # scaled_df = pd.DataFrame(scaled_data, columns=data_to_scale.columns)
    # full_scaled_df = pd.concat([first_column.reset_index(drop=True), scaled_df], axis=1)

    # 分割 connectors
    num_columns = len(target_df.columns)
    connector_len = (num_columns - 1) // 3

    connector1 = pd.concat(
        [first_column.reset_index(drop=True), target_df.iloc[:, 1:connector_len + 1]], axis=1
    )
    connector2 = pd.concat(
        [first_column.reset_index(drop=True), target_df.iloc[:, connector_len + 1:connector_len * 2 + 1]], axis=1
    )
    connector3 = pd.concat(
        [first_column.reset_index(drop=True), target_df.iloc[:, -connector_len:]], axis=1
    )

    # 按重要性计算加权距离
    connectors = {
        'connector1': connector1,
        'connector2': connector2,
        'connector3': connector3
    }

    return connectors

def calculate_euclidean_distances(query, df):
    """
    计算每个数据点与给定查询点的欧几里得距离。

    参数:
    - query: 查询点的 pdb_id。
    - df: 数据框，第1列为 pdb_id，其余列为数据特征。

    返回:
    - distances: 包含每个数据点与查询点的欧几里得距离的列表。
    """
    # 提取查询点的 embedding
    query_data = query.iloc[:, 1:].to_numpy()
    if query_data.shape[0] != 1:
        raise ValueError(f"Input query was None or had multiple rows\n{query}")
    
    # 提取其余点的 embedding
    data = df.iloc[:, 1:].to_numpy()

    # 计算距离
    distances = euclidean_distances(data, query_data)

    # 返回为一维数组
    return distances.flatten()


def calculate_cosine_similarities(query, df):
    """
    计算每个数据点与给定查询点的余弦相似性。

    参数:
    - query: 查询点的 pdb_id。
    - df: 数据框，第1列为 pdb_id，其余列为数据特征。

    返回:
    - similarities: 包含每个数据点与查询点的余弦相似性的 NumPy 一维数组。
      值域在 [-1, 1]，越接近 1 表示越相似。
    """
    # 提取查询点的 embedding
    query_data = query.iloc[:, 1:].to_numpy()
    if query_data.shape[0] != 1:
        raise ValueError(f"Input query was None or had multiple rows\n{query}")

    # 提取所有点的 embedding
    data = df.iloc[:, 1:].to_numpy()

    # 计算余弦相似性矩阵：结果形状为 (n_samples, 1)
    # sklearn.metrics.pairwise.cosine_similarity 在内部自动把分子（点积）除以两个向量的 L2 范数的乘积，从而“在计算相似度时”隐式地做了归一化。
    similarities = cosine_similarity(data, query_data)

    return similarities.flatten()

def calculate_weighted_distances(query_connectors, target_connectors, weights):
    """
    计算加权欧几里得距离和加权余弦相似性，并返回按欧几里得距离升序排列的 DataFrame。

    参数:
    - query_connectors: dict, {'connector1': DataFrame, 'connector2': ..., 'connector3': ...}
      每个 DataFrame 都只包含一行（query）。
    - target_connectors: dict, 同上，但每个 DataFrame 有 N 行（targets）。
    - weights: list of float, 长度为 3，对应三个 connector 的权重。

    返回:
    - output_df: pandas.DataFrame, 包含 ['pdb_id', 'euclidean', 'cosine']，按 euclidean 升序。
    """
    keys = ['connector1', 'connector2', 'connector3']
    euclid_dists = {}
    cosine_sims  = {}

    # 1) 对每个 connector 计算距离/相似性
    for key in keys:
        q_df = query_connectors[key]
        t_df = target_connectors[key]

        euclid_dists[key] = calculate_euclidean_distances(q_df, t_df)
        cosine_sims[key]  = calculate_cosine_similarities(q_df, t_df)

    # 2) 用第一个 connector 的长度来初始化累加向量
    n_samples = euclid_dists[keys[0]].shape[0]
    weighted_euclid = np.zeros(n_samples, dtype=float)
    weighted_cosine = np.zeros(n_samples, dtype=float)

    # 3) 加权累加
    for i, key in enumerate(keys):
        w = weights[i]
        weighted_euclid += euclid_dists[key] * w
        weighted_cosine += cosine_sims[key]  * w

    # 4) 拼回 pdb_id 并排序
    pdb_ids = target_connectors[keys[0]].iloc[:, 0].tolist()
    output_df = pd.DataFrame({
        'pdb_id':   pdb_ids,
        'euclidean': weighted_euclid,
        'cosine':    weighted_cosine
    })
    output_df = output_df.sort_values(by='euclidean', ascending=True).reset_index(drop=True)

    return output_df

def main(query_table_path, target_table_path, output_similarity_tsv):
    query_df = pd.read_csv(query_table_path, sep='\t')
    query_df.rename(columns={query_df.columns[0]: 'pdb_id'}, inplace=True)

    target_df = pd.read_csv(target_table_path, sep='\t')
    target_df.rename(columns={target_df.columns[0]: 'pdb_id'}, inplace=True)

    # 查找存在空缺值的行
    rows_with_nan = target_df[target_df.isnull().any(axis=1)]
    if not rows_with_nan.empty:
        # 提取出有 NaN 的 pdb_id 列表
        nan_pdb_ids = rows_with_nan['pdb_id'].tolist()
        # 打印这些 pdb_id
        print("Removed rows with NaN for pdb_id:", nan_pdb_ids)
        # 真正删除这些行
        target_df = target_df.drop(rows_with_nan.index).reset_index(drop=True)

    query_connectors = split_connector(query_df)
    target_connectors = split_connector(target_df)

    # weights = [0.2, 0.4, 0.4]  # connector1 : connector2 : connector3 = 0.2 : 0.2 : 0.4
    # output_df = calculate_weighted_distances(query_df, target_connectors, weights)
    # output_df.to_csv(output_similarity_tsv, sep='\t', index=False)

    # All data
    # 计算欧几里得距离
    euclid_dists_all = calculate_euclidean_distances(query_df, target_df)
    # 计算余弦相似性
    cosine_sims_all = calculate_cosine_similarities(query_df, target_df)

    pdb_ids = target_df.iloc[:, 0].to_list()
    all_output_df = pd.DataFrame({
        'pdb_id': pdb_ids,
        'euclidean': euclid_dists_all,
        'cosine': cosine_sims_all
    })
    # 按加权欧几里得距离升序排列
    all_output_df = all_output_df.sort_values(by='euclidean', ascending=True).reset_index(drop=True)
    all_output_df.to_csv(f'{os.path.splitext(output_similarity_tsv)[0]}_all.tsv' , sep='\t', index=False)

    # Obtain connectors distance individually
    weights_1 = [1, 0, 0]
    output_df_1 = calculate_weighted_distances(query_connectors, target_connectors, weights_1)
    output_similarity_tsv_1 = f'{os.path.splitext(output_similarity_tsv)[0]}_connector1.tsv' 
    output_df_1.to_csv(output_similarity_tsv_1, sep='\t', index=False)

    weights_2 = [0, 1, 0]
    output_df_2 = calculate_weighted_distances(query_connectors, target_connectors, weights_2)
    output_similarity_tsv_2 = f'{os.path.splitext(output_similarity_tsv)[0]}_connector2.tsv' 
    output_df_2.to_csv(output_similarity_tsv_2, sep='\t', index=False)

    weights_3 = [0, 0, 1]
    output_df_3 = calculate_weighted_distances(query_connectors, target_connectors, weights_3)
    output_similarity_tsv_3 = f'{os.path.splitext(output_similarity_tsv)[0]}_connector3.tsv' 
    output_df_3.to_csv(output_similarity_tsv_3, sep='\t', index=False)

