#!/usr/bin/env python3
import pandas as pd
import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Apply Kabat-position mutations to numbering table and output FASTA")
    p.add_argument("-t", "--target", required=True, help="Target CSV with numbering (first column 'Position')")
    p.add_argument("-m", "--mutation", required=True, help="Mutation CSV with columns 'Position','MutAA'")
    p.add_argument("-o", "--output", required=True, help="Output FASTA file")
    return p.parse_args()

def main():
    args = parse_args()
    # Read the target numbering table.
    try:
        df = pd.read_csv(args.target, dtype=str)
    except Exception as e:
        sys.exit(f"Error reading target CSV: {e}")

    if 'Position' not in df.columns:
        sys.exit("Error: target CSV must have a 'Position' column")

    # Read the mutation table.
    try:
        mut_df = pd.read_csv(args.mutation, dtype=str)
    except Exception as e:
        sys.exit(f"Error reading mutation CSV: {e}")

    if not {'Position','MutAA'}.issubset(mut_df.columns):
        sys.exit("Error: mutation CSV must have 'Position' and 'MutAA' columns")

    # For each mutation site, substitute the residue across all sequence columns.
    seq_cols = [c for c in df.columns if c != 'Position']
    df_mod = df.copy()
    for _, row in mut_df.iterrows():
        pos = row['Position']
        aa  = row['MutAA']
        mask = df_mod['Position'] == pos
        if not mask.any():
            sys.stderr.write(f"Warning: position {pos} not found in target table, skipping\n")
            continue
        # Replace every sequence entry in this row with the mutant amino acid.
        df_mod.loc[mask, seq_cols] = aa

    # Write FASTA: concatenate each column into a sequence, remove gap characters ('-'), and use the column name as the identifier.
    with open(args.output, 'w') as out_f:
        for col in seq_cols:
            seq = ''.join(df_mod[col].tolist()).replace('-', '')
            out_f.write(f">{col}\n")
            # Wrap the sequence at 70 characters per line.
            for i in range(0, len(seq), 70):
                out_f.write(seq[i:i+70] + "\n")

    print(f"Written mutated sequences to {args.output}")

if __name__ == "__main__":
    main()
