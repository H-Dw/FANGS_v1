#!/usr/bin/env python3
"""
extract_fr_kabat.py

Extract framework regions FR1, FR2, FR3, and FR4 from a CSV numbering table
exported by ANARCI under the Kabat scheme. The first column is Position, and
the remaining columns contain the individual sequences. Kabat numbering
intervals are applied for heavy or light chains, which may be specified
explicitly or inferred automatically, and the concatenated framework sequence
is written to FASTA.

Example:
  python extract_fr_kabat.py -i anarci_kabat.csv -o fr_kabat.fasta --chain-type auto

Arguments:
  --chain-type  heavy | light | auto  (default: auto)
  --keep-gaps   If set, retain alignment gaps ('-') in the output sequences;
                gaps are removed by default.
  --wrap N      FASTA line-wrap length (default: 60)
"""
import argparse
import pandas as pd
import re
import sys
from textwrap import wrap

# Kabat ranges used by this script (common / conservative definitions)
KABAT_RANGES = {
    'heavy': {
        'FR1': (1, 30),
        'CDR1': (31, 35),
        'FR2': (36, 49),
        'CDR2': (50, 65),
        'FR3': (66, 94),
        'CDR3': (95, 102),
        'FR4': (103, 113),
    },
    'light': {
        'FR1': (1, 23),
        'CDR1': (24, 34),
        'FR2': (35, 49),
        'CDR2': (50, 56),
        'FR3': (57, 88),
        'CDR3': (89, 97),
        'FR4': (98, 107),
    }
}

def parse_args():
    p = argparse.ArgumentParser(description="Extract Kabat FR1-4 sequences from ANARCI/Kabat CSV to FASTA")
    p.add_argument('-i','--input', required=True, help="Input CSV file (Position + sequence columns)")
    p.add_argument('-o','--output', required=True, help="Output FASTA file")
    p.add_argument('--chain-type', choices=('heavy','light','auto'), default='auto',
                   help="Chain type for Kabat ranges (default auto)")
    p.add_argument('--keep-gaps', action='store_true', help="Keep '-' gaps in returned FR sequences (default: remove gaps)")
    p.add_argument('--wrap', type=int, default=60, help="FASTA line wrap length (default 60)")
    return p.parse_args()

def parse_label(label):
    """Parse a position label like '111', '111A', '35b', '35.1' -> (int_idx, insertion_str)"""
    if pd.isna(label):
        return (None, '')
    s = str(label).strip()
    m = re.match(r'^(\d+)(.*)$', s)
    if not m:
        return (None, '')
    idx = int(m.group(1))
    ins = m.group(2) or ''
    return (idx, ins)

def detect_chain_type(df_idx):
    """
    Heuristic auto-detection:
    - if any index in heavy CDR3 interval (95-102) appears -> likely heavy
    - elif any index in light CDR3 interval (89-97) appears -> likely light
    - else default to heavy (and warn)
    """
    idxs = set([i for i in df_idx if i is not None])
    if any(95 <= i <= 102 for i in idxs):
        return 'heavy'
    if any(89 <= i <= 97 for i in idxs):
        return 'light'
    # fallback: check for presence of positions typical of heavy FR3 (66-94) vs light FR3 (57-88)
    if any(66 <= i <= 94 for i in idxs) and not any(57 <= i <= 88 for i in idxs):
        return 'heavy'
    if any(57 <= i <= 88 for i in idxs) and not any(66 <= i <= 94 for i in idxs):
        return 'light'
    return 'heavy'  # default

def collect_region(df_sorted, seq_col, start, end, keep_gaps=False):
    mask = (df_sorted['__idx__'].notnull()) & (df_sorted['__idx__'].between(start, end))
    rows = df_sorted.loc[mask, seq_col].fillna('-').tolist()
    if keep_gaps:
        return ''.join(rows)
    else:
        return ''.join([r for r in rows if r != '-'])

def write_fasta(records, outpath, wrap_len=60):
    with open(outpath, 'w') as fh:
        for hdr, seq in records:
            fh.write(f">{hdr}\n")
            for line in wrap(seq, wrap_len):
                fh.write(line + '\n')

def main():
    args = parse_args()
    try:
        df = pd.read_csv(args.input, dtype=str)
    except Exception as e:
        sys.stderr.write(f"Error reading input CSV: {e}\n")
        sys.exit(1)

    if 'Position' not in df.columns:
        sys.stderr.write("Error: expected first column named 'Position'.\n")
        sys.exit(1)

    # parse position labels
    positions = df['Position'].astype(str).tolist()
    parsed = [parse_label(lbl) for lbl in positions]
    df = df.copy()
    df['__idx__'] = [p[0] for p in parsed]
    df['__ins__'] = [p[1] for p in parsed]

    # sort by numeric index then insertion string (empty first)
    df_sorted = df.sort_values(by=['__idx__','__ins__'], 
                               key=lambda col: col.fillna('' if col.name=='__ins__' else 0))

    # columns corresponding to sequences (exclude Position, __idx__, __ins__)
    seq_cols = [c for c in df_sorted.columns if c not in ('Position','__idx__','__ins__')]

    # determine chain type
    chain_type = args.chain_type
    if chain_type == 'auto':
        inferred = detect_chain_type(df['__idx__'].tolist())
        if inferred != 'heavy':
            # if inferred light, set accordingly and print info
            chain_type = inferred
            print(f"Auto-detected chain type: {chain_type}")
        else:
            # if inferred heavy, still print
            print(f"Auto-detected chain type: {chain_type}")
            chain_type = inferred

    ranges = KABAT_RANGES[chain_type]

    records = []
    for seqid in seq_cols:
        fr_parts = []
        # order FR1, FR2, FR3, FR4 (skip CDRs)
        # For FR1 we collect from FR1 range; for FR2 from FR2; FR3 from FR3; FR4 from FR4
        fr1 = collect_region(df_sorted, seqid, ranges['FR1'][0], ranges['FR1'][1], keep_gaps=args.keep_gaps)
        fr2 = collect_region(df_sorted, seqid, ranges['FR2'][0], ranges['FR2'][1], keep_gaps=args.keep_gaps)
        fr3 = collect_region(df_sorted, seqid, ranges['FR3'][0], ranges['FR3'][1], keep_gaps=args.keep_gaps)
        fr4 = collect_region(df_sorted, seqid, ranges['FR4'][0], ranges['FR4'][1], keep_gaps=args.keep_gaps)
        concatenated = ''.join([fr1, fr2, fr3, fr4])
        header = f"{seqid}|Kabat|{chain_type}|FR1234"
        records.append((header, concatenated))

    # write FASTA
    try:
        write_fasta(records, args.output, wrap_len=args.wrap)
    except Exception as e:
        sys.stderr.write(f"Error writing FASTA: {e}\n")
        sys.exit(1)

    print(f"Wrote {len(records)} FR1234 (Kabat, {chain_type}) sequences to {args.output}")

if __name__ == '__main__':
    main()
