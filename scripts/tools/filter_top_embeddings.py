import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Filter top 10% entries by original euclidean embeddings and merge with grafted embeddings."
    )
    parser.add_argument("--input", required=True, help="Input folder containing the two TSV files")
    parser.add_argument("--output", required=True, help="Output file path for the merged filtered table (TSV)")
    parser.add_argument("--orig_col", default="original_euclidean_all_raw_embeddings", 
                   help="Column name for original embeddings (default: original_euclidean_all_raw_embeddings)")
    parser.add_argument("--graf_col", default="grafted_euclidean_all_raw_embeddings",
                    help="Column name for grafted embeddings (default: grafted_euclidean_all_raw_embeddings)")

    args = parser.parse_args()

    input_folder = args.input
    output_path = args.output

    # --- File paths ---
    original_path = os.path.join(input_folder, "original_raw_embeddings.tsv")
    grafted_path = os.path.join(input_folder, "grafted_raw_embeddings.tsv")

    # --- Load tables ---
    print(f"Reading original embeddings from: {original_path}")
    original_df = pd.read_csv(original_path, sep="\t")

    print(f"Reading grafted embeddings from: {grafted_path}")
    grafted_df = pd.read_csv(grafted_path, sep="\t")

    # --- Keep only required columns ---
    original_col = args.orig_col
    grafted_col = args.graf_col

    pdb_col = "PDB_ID"

    original_df = original_df[[pdb_col, original_col]].copy()
    grafted_df = grafted_df[[pdb_col, grafted_col]].copy()

    print(f"Original table shape: {original_df.shape}")
    print(f"Grafted table shape:  {grafted_df.shape}")

    # --- Filter top 10% rows by original_euclidean_all_raw_embeddings ---
    # "Top 10%" means the lowest 10% of euclidean distance values (smallest distances = best matches)
    threshold = original_df[original_col].quantile(0.10)
    filtered_original = original_df[original_df[original_col] <= threshold].copy()

    print(f"Top 10% threshold (original euclidean): {threshold:.6f}")
    print(f"Rows after filtering: {len(filtered_original)} / {len(original_df)}")

    # --- Extract matching rows from grafted table ---
    selected_pdb_ids = filtered_original[pdb_col].unique()
    filtered_grafted = grafted_df[grafted_df[pdb_col].isin(selected_pdb_ids)].copy()
    filtered_grafted = filtered_grafted.sort_values(by=grafted_col, ascending=True).reset_index(drop=True)

    # --- Round numerical columns to 2 decimal places ---
    filtered_original[original_col] = filtered_original[original_col].round(2)
    filtered_grafted[grafted_col] = filtered_grafted[grafted_col].round(2)

    print(f"Matching grafted rows found: {len(filtered_grafted)}")

    # --- Merge the two filtered tables on PDB_ID ---
    if original_col == grafted_col:
        print("Original and grafted column names are the same, renaming grafted column to avoid conflict...")
        filtered_grafted.rename(columns={grafted_col: grafted_col + "_grafted"}, inplace=True)
        filtered_original.rename(columns={original_col: original_col + "_original"}, inplace=True)
        
    merged_df = pd.merge(filtered_grafted, filtered_original, on=pdb_col, how="inner")

    print(f"Merged table shape: {merged_df.shape}")

    # # --- Sort by grafted_euclidean_all_raw_embeddings ascending ---
    # merged_df = merged_df.sort_values(by=grafted_col, ascending=True).reset_index(drop=True)

    # --- Save output ---
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merged_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved merged filtered table to: {output_path}")


if __name__ == "__main__":
    main()
