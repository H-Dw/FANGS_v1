import pandas as pd
import os

def transformat(pdb_cdr_df, pdb_path, output_file):
    # 生成 pdb_path + 文件名
    pdb_path = os.path.abspath(pdb_path)
    # pdb_cdr_df['pdb_path'] = pdb_cdr_df['pdb_id'].apply(lambda x: f"{pdb_path}/{x}.pdb")
    pdb_cdr_df['pdb_path'] = pdb_cdr_df['target'].apply(
        lambda x: f"{pdb_path}/{x}" if x.endswith('.pdb') else f"{pdb_path}/{x}.pdb"
    )
    
    # 从 target 列提取 chain (假设为第 4 个字符)
    # pdb_cdr_df['chain'] = pdb_cdr_df['pdb_id'].str[4:]
    pdb_cdr_df['chain'] = pdb_cdr_df['target'].str[4:]
    # NOTE: generated false formate
    
    # 计算 CDR3 序列的长度
    pdb_cdr_df['cdr3_length'] = pdb_cdr_df['cdr3_seq'].apply(len)
    
    # 构建所需的列并保存为 TSV 文件
    output_df = pdb_cdr_df[['pdb_path', 'chain', 'cdr1_seq', 'cdr2_seq', 'cdr3_seq', 'cdr3_length']]
    output_df.columns = ['pdb_path', 'chain', 'cdr1', 'cdr2', 'cdr3', 'cdr3_length']  # 重命名列
    output_df.to_csv(output_file, sep='\t', index=False)
    return output_df

# def extract_top(order_df, pdb_cdr_file, top_n, output_file):
#     pdb_cdr_df = pd.read_csv(pdb_cdr_file, sep='\t')
#     pdb_cdr_df['target'] = pdb_cdr_df['target'].str.replace(r"\.pdb$", "", regex=True)

#     # Extract top N row
#     top_n = int(top_n)
#     extracted_df = order_df.iloc[:top_n]

#     set_extracted = set(extracted_df['pdb_id'])
#     set_targets  = set(pdb_cdr_df['target'])
#     missing = set_extracted - set_targets
#     print(f"{len(missing)} of top {top_n} pdb_id not found in pdb_cdr_df. \n {list(missing)}")

#     top_df = pd.merge(extracted_df, pdb_cdr_df, how='inner', left_on='pdb_id', right_on='target')

#     # top_df.to_csv(f"{os.path.dirname(output_file)}/extract_top.tsv", sep='\t')
#     return top_df

def extract_top(order_df, pdb_cdr_file, top_n, dupl_pdb=False):
    pdb_cdr_df = pd.read_csv(pdb_cdr_file, sep='\t')
    pdb_cdr_df['target'] = pdb_cdr_df['target'].str.replace(r"\.pdb$", "", regex=True)

    # duplicates pdb
    if dupl_pdb:
        order_df['PDB'] = order_df['pdb_id'].str[:4]
        order_df = order_df.drop_duplicates(subset='PDB', keep='first').reset_index(drop=True)

    set_order= set(order_df['pdb_id'])
    set_targets  = set(pdb_cdr_df['target'])
    missing = set_order - set_targets

    top_n = int(top_n)
    num_order = len(order_df)
    top_n = num_order if top_n == 0 else top_n

    if len(missing) > 0:
        print(f"{len(missing)} of order {top_n} ID not found in pdb_cdr_df. \n Missing: {list(missing)}")

    merged_df = pd.merge(order_df, pdb_cdr_df, how='inner', left_on='pdb_id', right_on='target')

    # Extract top N row
    top_df = merged_df.iloc[:top_n]

    # top_df.to_csv(f"{os.path.dirname(output_file)}/extract_top.tsv", sep='\t')
    return top_df

def deduplication(file, output_file):
    df = pd.read_csv(file, sep='\t')

    df['pdb_id'] = df['pdb_id'].str.replace(r'\.pdb$', '', regex=True)    # Select the PDB id (removed chain id)
    df_unique = df.drop_duplicates(subset=['pdb_id','tSeq'])   # Avoid those PDB contain different Nb

    # output_unique_file = f"{os.path.dirname(output_file)}/unique_order_file.tsv"
    # df_unique.to_csv(output_unique_file, sep='\t', index=True)

    return df_unique
    
def main(distance_file, pdb_cdr_file, pdb_path, top_n, output_file):
    distance_unique_df = deduplication(distance_file, output_file)
    merged_df = extract_top(distance_unique_df, pdb_cdr_file, top_n)
    output_df = transformat(merged_df, pdb_path, output_file)
    return output_df
