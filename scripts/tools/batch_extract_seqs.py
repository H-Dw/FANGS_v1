import os
import glob
import argparse
import sys
from typing import Optional

def extract_sequence_from_pdb(file_path: str) -> Optional[str]:
    """
    Extract the amino-acid sequence from a PDB file and return a standardized FASTA record.

    Args:
        file_path: Path to the PDB file.

    Returns:
        A FASTA-formatted string, or None if extraction fails.
    """
    try:
        from readPDBSeq import get_sequence_from_pdb
        sequence = get_sequence_from_pdb(file_path)
        if not sequence:
            return None
            
        header = os.path.splitext(os.path.basename(file_path))[0]
        return f">{header}\n{sequence}"
        
    except ImportError:
        print("Error: the readPDBSeq module was not found; ensure that it is installed", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error while processing {file_path}: {e}", file=sys.stderr)
        return None

def main(db_path: str, output_file: str) -> None:
    """
    Extract sequences from PDB files and write them to a FASTA file.

    Args:
        db_path: Input path, given as a glob pattern or a directory.
        output_file: Path to the output FASTA file.
    """
    # Validate the input path.
    if not os.path.exists(db_path):
        print(f"Input path does not exist: {db_path}", file=sys.stderr)
        return

    # Assemble the list of PDB files.
    if os.path.isdir(db_path):
        pattern = os.path.join(db_path, "*.pdb")
    else:
        pattern = db_path
        
    pdb_files = glob.glob(pattern)
    if not pdb_files:
        print(f"No PDB files matched the pattern: {pattern}", file=sys.stderr)
        return

    # Extract sequences and write the FASTA output.
    try:
        with open(output_file, 'w') as f_out:
            for file in pdb_files:
                fasta_entry = extract_sequence_from_pdb(file)
                if fasta_entry:
                    f_out.write(fasta_entry + '\n')
        print(f"Processed {len(pdb_files)} files; output written to {output_file}")
                    
    except IOError as e:
        print(f"Failed to write the output file: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract protein sequences from PDB files and write them in FASTA format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("db_path", help="PDB path (glob pattern) or a directory containing PDB files")
    parser.add_argument("output_file", help="Path to the output FASTA file")
    
    args = parser.parse_args()
    
    # Optionally prepend the readPDBSeq module directory to sys.path.
    # sys.path.append('/path/to/readPDBSeq')
    
    main(args.db_path, args.output_file)