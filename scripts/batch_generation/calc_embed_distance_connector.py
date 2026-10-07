import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import os

def split_connector(target_df):
    first_column = target_df.iloc[:, 0]  # Identifier column.
    data_to_scale = target_df.iloc[:, 1:]  # Remaining columns, containing the embedding features.
    
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(data_to_scale)
    # scaled_df = pd.DataFrame(scaled_data, columns=data_to_scale.columns)
    # full_scaled_df = pd.concat([first_column.reset_index(drop=True), scaled_df], axis=1)

    # Partition the embedding columns into three connector blocks.
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

    # Group the connector blocks for subsequent weighted-distance calculation.
    connectors = {
        'connector1': connector1,
        'connector2': connector2,
        'connector3': connector3
    }

    return connectors

def calculate_euclidean_distances(query, df):
    """
    Compute the Euclidean distance from one query embedding to each target embedding.

    Parameters
    ----------
    query : pandas.DataFrame
        Query table containing a single row. Column 1 stores the PDB identifier;
        the remaining columns store embedding features.
    df : pandas.DataFrame
        Target table. Column 1 stores the PDB identifier; the remaining columns
        store embedding features.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of Euclidean distances, with one entry per target row.
    """
    # Extract the query embedding.
    query_data = query.iloc[:, 1:].to_numpy()
    if query_data.shape[0] != 1:
        raise ValueError(f"Input query was None or had multiple rows\n{query}")
    
    # Extract the target embeddings.
    data = df.iloc[:, 1:].to_numpy()

    # Compute pairwise Euclidean distances.
    distances = euclidean_distances(data, query_data)

    # Return a one-dimensional array.
    return distances.flatten()


def calculate_cosine_similarities(query, df):
    """
    Compute the cosine similarity between one query embedding and each target embedding.

    Parameters
    ----------
    query : pandas.DataFrame
        Query table containing a single row. Column 1 stores the PDB identifier;
        the remaining columns store embedding features.
    df : pandas.DataFrame
        Target table. Column 1 stores the PDB identifier; the remaining columns
        store embedding features.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of cosine similarities on [-1, 1]. Values closer
        to 1 indicate greater similarity.
    """
    # Extract the query embedding.
    query_data = query.iloc[:, 1:].to_numpy()
    if query_data.shape[0] != 1:
        raise ValueError(f"Input query was None or had multiple rows\n{query}")

    # Extract embeddings for every row.
    data = df.iloc[:, 1:].to_numpy()

    # Cosine-similarity matrix of shape (n_samples, 1).
    # sklearn.metrics.pairwise.cosine_similarity divides each dot product by the
    # product of the two L2 norms, thereby normalizing both vectors during the comparison.
    similarities = cosine_similarity(data, query_data)

    return similarities.flatten()

def calculate_weighted_distances(query_connectors, target_connectors, weights):
    """
    Compute weighted Euclidean distances and weighted cosine similarities.

    Parameters
    ----------
    query_connectors : dict
        Mapping {'connector1': DataFrame, 'connector2': DataFrame, 'connector3': DataFrame}.
        Each DataFrame contains the query as a single row.
    target_connectors : dict
        Mapping with the same keys. Each DataFrame contains N target rows.
    weights : list of float
        Three weights, one for each connector, in connector order.

    Returns
    -------
    pandas.DataFrame
        Columns ['pdb_id', 'euclidean', 'cosine'], sorted by Euclidean distance
        in ascending order.
    """
    keys = ['connector1', 'connector2', 'connector3']
    euclid_dists = {}
    cosine_sims  = {}

    # 1) Compute the distance and similarity of each connector.
    for key in keys:
        q_df = query_connectors[key]
        t_df = target_connectors[key]

        euclid_dists[key] = calculate_euclidean_distances(q_df, t_df)
        cosine_sims[key]  = calculate_cosine_similarities(q_df, t_df)

    # 2) Initialize the accumulators to the length of the first connector.
    n_samples = euclid_dists[keys[0]].shape[0]
    weighted_euclid = np.zeros(n_samples, dtype=float)
    weighted_cosine = np.zeros(n_samples, dtype=float)

    # 3) Accumulate the weighted distances and similarities.
    for i, key in enumerate(keys):
        w = weights[i]
        weighted_euclid += euclid_dists[key] * w
        weighted_cosine += cosine_sims[key]  * w

    # 4) Reattach the PDB identifiers and sort the table.
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

    # Identify rows that contain missing values.
    rows_with_nan = target_df[target_df.isnull().any(axis=1)]
    if not rows_with_nan.empty:
        # Collect the PDB identifiers of rows that contain NaN.
        nan_pdb_ids = rows_with_nan['pdb_id'].tolist()
        # Report the removed identifiers.
        print("Removed rows with NaN for pdb_id:", nan_pdb_ids)
        # Remove those rows from the target table.
        target_df = target_df.drop(rows_with_nan.index).reset_index(drop=True)

    query_connectors = split_connector(query_df)
    target_connectors = split_connector(target_df)

    # weights = [0.2, 0.4, 0.4]  # connector1 : connector2 : connector3 = 0.2 : 0.2 : 0.4
    # output_df = calculate_weighted_distances(query_df, target_connectors, weights)
    # output_df.to_csv(output_similarity_tsv, sep='\t', index=False)

    # All data
    # Compute Euclidean distances over the full embedding.
    euclid_dists_all = calculate_euclidean_distances(query_df, target_df)
    # Compute cosine similarities over the full embedding.
    cosine_sims_all = calculate_cosine_similarities(query_df, target_df)

    pdb_ids = target_df.iloc[:, 0].to_list()
    all_output_df = pd.DataFrame({
        'pdb_id': pdb_ids,
        'euclidean': euclid_dists_all,
        'cosine': cosine_sims_all
    })
    # Sort by Euclidean distance in ascending order.
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

