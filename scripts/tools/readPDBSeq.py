from Bio.PDB import PDBParser

# Dictionary to map three-letter amino acid codes to one-letter codes
three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

non_standard_three_to_one = {
    'MSE': 'M',  # 硒代蛋氨酸 (Selenomethionine)
    'SEP': 'S',  # 磷酸丝氨酸 (Phosphoserine)
    'TPO': 'T',  # 磷酸苏氨酸 (Phosphothreonine)
    'PTR': 'Y',  # 磷酸酪氨酸 (Phosphotyrosine)
    'HYP': 'O',  # 羟脯氨酸 (Hydroxyproline)
    'HYL': 'K',  # 羟赖氨酸 (Hydroxylysine)
    'MLZ': 'K',  # 甲基赖氨酸 (Methyllysine)
    'ALY': 'K',  # 乙酰赖氨酸 (Acetyllysine)
    'CSS': 'C',  # 半胱氨酸二硫键 (Cystine)
    'PCA': 'X',  # 环化脯氨酸 (Pyroglutamate)
    'NLE': 'N',  # 异硬蛋白氨酸 (Norleucine)
    'FTY': 'Y',  # 氟代酪氨酸 (Fluorotyrosine)
    'AZF': 'F',  # 偶氮苯丙氨酸 (Azidophenylalanine)
    'NPF': 'F',  # 硝基苯丙氨酸 (Nitrophenylalanine)
    'PYL': 'O',  # 杂环赖氨酸 (Pyrrolysine)
    'CSO': 'C'   # 氧化半胱氨酸 (Oxidized Cysteine)
}


def get_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                # 获取每个氨基酸的三字母缩写
                res_name = residue.get_resname()
                # 将三字母缩写转换为标准的氨基酸单字母代码
                one_letter_code = three_to_one.get(res_name)
                if one_letter_code:
                    sequence += one_letter_code
                elif one_letter_code is None:
                    # 查找非天然氨基酸
                    one_letter_code = non_standard_three_to_one.get(res_name)
                    if one_letter_code is not None:
                        # print(f'Searched non standard amino acid: {res_name}')
                        sequence += one_letter_code
                else:
                    # 如果无法找到对应的氨基酸单字母代码，则跳过
                    print(f"Warming: '{res_name}' counld not be identified in amino acid table !")
                    continue

    return sequence

def find_sequence_location(full_sequence, sub_sequence):
    start_index = full_sequence.find(sub_sequence)
    if start_index != -1:
        start_index += 1
        end_index = start_index + len(sub_sequence) - 1
        return start_index, end_index
    else:
        return None, None

def find_seq_loc_from_pdb(pdb_file, query):

    # 定义完整序列和子序列
    # 调用函数获取PDB文件中的氨基酸序列
    full_sequence = get_sequence_from_pdb(pdb_file)
    sub_sequence = query

    # 调用函数查找子序列的位置
    start, end = find_sequence_location(full_sequence, sub_sequence)

    return start, end

def main(pdb_file, cdr_seq):
    cdr_start = 0
    cdr_end = 0
    seq_length = 0
    try:
        full_sequence = get_sequence_from_pdb(pdb_file)
        # 获取序列总长度
        seq_length = len(full_sequence)
        cdr_start, cdr_end = find_seq_loc_from_pdb(pdb_file, cdr_seq)
        # 输出格式为：起始位置 CDR长度 总序列长度
        print(f"{cdr_start} {cdr_end} {seq_length}")
    except ValueError as e:
        print(e)
    return cdr_start, cdr_end, seq_length

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python readPDBSeq.py <pdb_file> <cdr_seq>")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    cdr_seq = sys.argv[2]
    
    cdr_start, cdr_end, seq_length = main(pdb_file, cdr_seq)