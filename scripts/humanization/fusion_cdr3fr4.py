import argparse
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def parse_args():
    parser = argparse.ArgumentParser(
        description='Append target sequence to each DB sequence, filtering out those with internal stop codons (*)')
    parser.add_argument('-d', '--db', required=True,
                        help='Input FASTA file containing DB sequences (db.fasta)')
    parser.add_argument('-t', '--target', required=True,
                        help='Input FASTA file containing target sequence (target.fasta)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output FASTA file for new sequences')
    return parser.parse_args()


def main():
    args = parse_args()

    # Read the target sequence (take the first record)
    target_records = list(SeqIO.parse(args.target, 'fasta'))
    if not target_records:
        raise ValueError(f"No sequences found in target file: {args.target}")
    target_seq = target_records[0].seq
    target_id = target_records[0].id

    # Process DB sequences
    new_records = []
    for record in SeqIO.parse(args.db, 'fasta'):
        seq_str = str(record.seq)
        # Skip sequences containing internal stop codon indicator '*'
        if '*' in seq_str:
            continue
        # Append target sequence
        appended_seq = Seq(seq_str + str(target_seq))
        # Create new SeqRecord
        db_id = record.id.split("|")[0]
        db_class = record.id.split("|")[1].replace("*", "_")
        new_id = f"{db_id}_{db_class}_{target_id}"
        new_record = SeqRecord(appended_seq, id=new_id, description='')
        new_records.append(new_record)

    # Write output
    with open(args.output, 'w') as out_handle:
        SeqIO.write(new_records, out_handle, 'fasta')

    print(f"Written {len(new_records)} sequences to {args.output}")


if __name__ == '__main__':
    main()
