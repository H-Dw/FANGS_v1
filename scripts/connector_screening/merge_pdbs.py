#!/usr/bin/env python3
"""
Merge multiple PDB files into a single PDB using Biopython,
without using StructureBuilder (avoids segid AttributeError).

Behavior:
 - All chains are converted to chain 'A'
 - Residue numbering is renumbered consecutively from 1
 - Atom serial numbers are renumbered consecutively from 1
 - Coordinates, occupancy, bfactor, altloc, element are preserved
 - HETATM/water are included (treated as standard residues with hetflag ' ')
"""

import argparse
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom


def merge_pdbs(pdb_files, output_file="merged.pdb"):
    merged_lines = []
    atom_id = 1
    res_id = 1

    for pdb_path in pdb_files:
        with open(pdb_path, "r") as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue

                # Parse the fixed-width PDB record fields.
                record = line[:6]
                atom_name = line[12:16]
                alt_loc = line[16]
                res_name = line[17:20]
                chain_id = "A"   # Assign a uniform chain identifier.
                x = line[30:38]
                y = line[38:46]
                z = line[46:54]
                occ = line[54:60]
                bfact = line[60:66]
                element = line[76:78]

                # 每遇到新的 residue，重排 residue ID
                # 使用原 residue ID 判断是否需要递增
                orig_res_seq = int(line[22:26].strip())
                if len(merged_lines) == 0 or orig_res_seq != last_orig_res_seq:
                    global_res_seq = res_id
                    res_id += 1

                last_orig_res_seq = orig_res_seq

                # Reformat the ATOM/HETATM record according to the PDB version 3 column layout.
                new_line = (
                    f"{record:<6}{atom_id:>5} "
                    f"{atom_name:<4}{alt_loc:<1}"
                    f"{res_name:>3} {chain_id}"
                    f"{global_res_seq:>4}    "
                    f"{x}{y}{z}{occ}{bfact}          "
                    f"{element:>2}"
                )

                merged_lines.append(new_line + "\n")
                atom_id += 1

    # Write the merged coordinate records to the output file.
    with open(output_file, "w") as f:
        f.write("".join(merged_lines))

    print(f"✔ 合并完成：{output_file}")
    print(f"✔ 共 {atom_id-1} 个原子，{res_id-1} 个残基")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple PDB files into one PDB. Chain -> 'A', residues+atoms renumbered."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output merged PDB filename"
    )
    parser.add_argument(
        "pdb_files",
        nargs="+",
        help="Input PDB files to merge (order preserved)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    merge_pdbs(args.pdb_files, args.output)


if __name__ == "__main__":
    main()
