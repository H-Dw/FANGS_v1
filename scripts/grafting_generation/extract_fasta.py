import csv, argparse
from collections import OrderedDict

def tsv_to_fasta(tsv_file, fasta_pdb_output, fasta_filename_output, prefix=""):
    pdb_sequences = OrderedDict()  # Preserve insertion order while collapsing duplicate PDB identifiers.
    filename_sequences = []

    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            pdb_id = row['PDB_ID']
            sequence = row['Sequence']
            filename = row['Filename']

            # Retain the first sequence observed for each PDB_ID.
            if pdb_id not in pdb_sequences:
                pdb_sequences[pdb_id] = sequence

            filename_sequences.append((filename, sequence))

    # Write a FASTA file whose headers are PDB identifiers.
    with open(fasta_pdb_output, 'w') as f:
        for pdb_id, seq in pdb_sequences.items():
            f.write(f">{prefix}{pdb_id}\n{seq}\n")

    # Write a FASTA file whose headers are source filenames.
    with open(fasta_filename_output, 'w') as f:
        for filename, seq in filename_sequences:
            f.write(f">{prefix}{filename}\n{seq}\n")

def main():
    parser = argparse.ArgumentParser(description="Build FASTA file based on generation")
    parser.add_argument('tsv_file', help="Path to TSV file listing generation information")
    parser.add_argument('fasta_pdb_output', default='pdb_id.fasta', help="fasta_pdb_output")
    parser.add_argument('fasta_filename_output', default='filename.fasta', help="fasta_filename_output")
    parser.add_argument('--prefix', default='', help="prefix")
    args = parser.parse_args()

    tsv_to_fasta(args.tsv_file, args.fasta_pdb_output, args.fasta_filename_output, args.prefix)

if __name__ == '__main__':
    main()