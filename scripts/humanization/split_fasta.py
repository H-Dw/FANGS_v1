#!/usr/bin/env python3
"""
split_fasta.py

Read a FASTA file from the command line, write each sequence to an individual
file, and place the resulting files in the specified output directory.

Usage:
    python split_fasta.py -i input.fasta -o output_dir
"""
import os
import argparse
from Bio import SeqIO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a FASTA file into individual sequence files and place them in the specified output directory."
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help="Path to the input FASTA file"
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help="Output directory, will be created if it does not exist"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    output_dir = args.output

    # Create the output directory if it does not already exist.
    os.makedirs(output_dir, exist_ok=True)

    # Parse the input and write each sequence.
    for record in SeqIO.parse(input_path, "fasta"):
        # Use the sequence identifier as the filename and retain the FASTA extension.
        filename = f"{record.id}.fasta"
        out_path = os.path.join(output_dir, filename)
        record.id = "A|protein|"
        record.description = ''

        # Write a single sequence.
        with open(out_path, "w") as handle:
            SeqIO.write(record, handle, "fasta")
        # print(f"Wrote: {out_path}")

if __name__ == '__main__':
    main()
