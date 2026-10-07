import pandas as pd
import sys, os

if len(sys.argv) != 3:
    print("Usage: python dedup_INDI_info.py <INDI_info_csv> <output_path>")
    sys.exit(1)

table_path = sys.argv[1]
output_path = sys.argv[2]

if os.path.isdir(output_path):
    print(f"Error: {output_path} is a directory, please provide a valid file path.")
    sys.exit(1)

df = pd.read_csv(table_path)

# Deduplicate the rows by Sequence, CDR1, CDR2, CDR3, and merge the PDBChain information
df_cleaned = df.groupby(['Sequence', 'CDR1', 'CDR2', 'CDR3']).agg(
    Member=('PDBChain', lambda x: ','.join(x)),
).reset_index()

# Add the merged member as a new column "member", and keep the original PDBChain column
df_cleaned = pd.merge(df.drop_duplicates(subset=['Sequence', 'CDR1', 'CDR2', 'CDR3']), df_cleaned, on=['Sequence', 'CDR1', 'CDR2', 'CDR3'])

df_cleaned.to_csv(output_path, index=False, sep='\t')

print(f"File saved to {output_path}")
