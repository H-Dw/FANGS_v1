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
    'MSE': 'M',  # selenomethionine
    'SEP': 'S',  # phosphoserine
    'TPO': 'T',  # phosphothreonine
    'PTR': 'Y',  # phosphotyrosine
    'HYP': 'O',  # hydroxyproline
    'HYL': 'K',  # hydroxylysine
    'MLZ': 'K',  # methyllysine
    'ALY': 'K',  # acetyllysine
    'CSS': 'C',  # cystine
    'PCA': 'X',  # pyroglutamate
    'NLE': 'N',  # norleucine
    'FTY': 'Y',  # fluorotyrosine
    'AZF': 'F',  # azidophenylalanine
    'NPF': 'F',  # nitrophenylalanine
    'PYL': 'O',  # pyrrolysine
    'CSO': 'C'   # oxidized cysteine
}


def get_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                # Three-letter residue name.
                res_name = residue.get_resname()
                # Map the three-letter code to the standard one-letter amino-acid code.
                one_letter_code = three_to_one.get(res_name)
                if one_letter_code:
                    sequence += one_letter_code
                elif one_letter_code is None:
                    # Look up a non-standard residue.
                    one_letter_code = non_standard_three_to_one.get(res_name)
                    if one_letter_code is not None:
                        # print(f'Searched non standard amino acid: {res_name}')
                        sequence += one_letter_code
                else:
                    # Skip residues that cannot be mapped to a one-letter code.
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

    # Define the full sequence and the query subsequence.
    # Retrieve the amino-acid sequence from the PDB file.
    full_sequence = get_sequence_from_pdb(pdb_file)
    sub_sequence = query

    # Locate the query subsequence.
    start, end = find_sequence_location(full_sequence, sub_sequence)

    return start, end

def main(pdb_file, cdr_seq):
    cdr_start = 0
    cdr_end = 0
    seq_length = 0
    try:
        full_sequence = get_sequence_from_pdb(pdb_file)
        # Total sequence length.
        seq_length = len(full_sequence)
        cdr_start, cdr_end = find_seq_loc_from_pdb(pdb_file, cdr_seq)
        # Output format: start position, CDR length, and total sequence length.
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