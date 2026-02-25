#!/usr/bin/env python3
"""
Renumber antibody sequences in FASTA using ANARCI (Kabat scheme) and output a CSV table.
The CSV has:
 - first column: residue position labels (e.g., 27, 27A, 27B...)
 - subsequent columns: amino acid at that position for each input sequence
Supports multiple sequences: gathers all unique position labels across sequences,
sorted by numeric index then insertion letter, and fills gaps with '-'.
Also reports total input sequences, number successfully numbered, and lists IDs of sequences
where all positions are gaps ("-").
"""
import argparse
import sys
import os
import csv

try:
    from anarci import run_anarci, read_fasta
except ImportError as e:
    sys.stderr.write(f"Fatal Error: {e}\n")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ANARCI Kabat renumbering to CSV summary table")
    parser.add_argument('-i','--input', required=True,
                        help="Input FASTA file")
    parser.add_argument('-o','--output', required=True,
                        help="Output CSV file path")
    parser.add_argument('-p','--processes', type=int, default=1,
                        help="Number of parallel processes")
    return parser.parse_args()


def label_from_position(pos):
    """Convert (index, insertion) tuple to string label"""
    idx, ins = pos
    return f"{idx}{ins if ins else ''}"


def main():
    args = parse_args()

    # Read input FASTA
    if not os.path.isfile(args.input):
        sys.stderr.write(f"Error: input file {args.input} not found\n")
        sys.exit(1)
    seqs = read_fasta(args.input)
    total_input = len(seqs)
    if total_input == 0:
        sys.stderr.write("Error: no sequences found in FASTA\n")
        sys.exit(1)

    # Run ANARCI with Kabat
    allow = set(['H','K','L'])
    try:
        sequences, numbered, _, _ = run_anarci(
            seqs,
            scheme='k',
            ncpu=args.processes,
            allow=allow,
            output=False
        )
    except Exception as e:
        sys.stderr.write(f"ANARCI error: {e}\n")
        sys.exit(1)

    # Build mapping: for each sequence, a dict label->aa and track stats
    seq_labels = []  # list of dicts for CSV
    names = []       # sequence IDs for CSV
    all_positions = set()
    drop_ids = []    # IDs where all residues are '-'
    success_count = 0

    for (name, _), domains in zip(sequences, numbered):
        label_map = {}
        if domains:
            success_count += 1
            for numbering, start, end in domains:
                for (idx, ins), aa in numbering:
                    label = label_from_position((idx, ins))
                    label_map[label] = aa
                    all_positions.add((idx, ins))
        # Determine if sequence has any real residue
        if not label_map:
            drop_ids.append(name)
            continue
        # Else include in CSV
        names.append(name)
        seq_labels.append(label_map)

    # Sort all_positions by idx then insertion order (empty before letters)
    def sort_key(pos):
        idx, ins = pos
        return (idx, ins or '')
    sorted_positions = sorted(all_positions, key=sort_key)

    # Write CSV
    try:
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow(['Position'] + names)
            # Rows
            for pos in sorted_positions:
                label = label_from_position(pos)
                row = [label]
                for label_map in seq_labels:
                    row.append(label_map.get(label, '-'))
                writer.writerow(row)
    except IOError as e:
        sys.stderr.write(f"Error writing CSV: {e}\n")
        sys.exit(1)

    # Summary
    print(f"Total input sequences: {total_input}")
    print(f"Successfully numbered sequences: {success_count}")
    if drop_ids:
        print("Sequences dropped (all gaps):")
        for did in drop_ids:
            print(f" - {did}")
    print(f"CSV numbering table written to {args.output}")

if __name__ == '__main__':
    main()
