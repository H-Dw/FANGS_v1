import argparse
import os
import pandas as pd

def process_single_tsv(input_path, output_path, index_list):
    df = pd.read_csv(input_path, sep="\t")

    # Find columns that match any index substring
    selected_cols = []
    for idx in index_list:
        idx = str(idx)
        matched = [c for c in df.columns if idx in c]
        selected_cols.extend(matched)

    selected_cols = list(set(selected_cols))  # Remove duplicates

    if not selected_cols:
        print(f"[Warning] No matching columns found in {os.path.basename(input_path)}. File skipped.")
        return

    # ----- NEW: split selected columns into euclidean / cosine -----
    euclidean_cols = [c for c in selected_cols if "euclidean" in c.lower()]
    cosine_cols = [c for c in selected_cols if "cosine" in c.lower()]

    # Create new average columns (if columns exist)
    if euclidean_cols:
        df["selected_euclidean_average"] = df[euclidean_cols].mean(axis=1)
    else:
        df["selected_euclidean_average"] = float("nan")

    if cosine_cols:
        df["selected_cosine_average"] = df[cosine_cols].mean(axis=1)
    else:
        df["selected_cosine_average"] = float("nan")

    # ----- Insert new columns as 2nd and 3rd columns -----
    cols = df.columns.tolist()
    cols.remove("selected_euclidean_average")
    cols.remove("selected_cosine_average")

    # Insert in order: index 1 → euclidean, index 2 → cosine
    cols.insert(1, "selected_euclidean_average")
    cols.insert(2, "selected_cosine_average")

    df = df[cols]

    # Sort by euclidean average (or cosine if desired)
    df = df.sort_values("selected_euclidean_average", ascending=True)

    # Save the processed file
    df.to_csv(output_path, sep="\t", index=False)

    print(f"Processed: {os.path.basename(input_path)}  →  {output_path}")


def process_tsv_folder(input_dir, output_dir, index_list):
    os.makedirs(output_dir, exist_ok=True)

    tsv_files = [f for f in os.listdir(input_dir) if f.endswith(".tsv")]

    if not tsv_files:
        print("No TSV files found in the input directory.")
        return

    for filename in tsv_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        process_single_tsv(input_path, output_path, index_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process each TSV file individually based on index-matched columns.")
    parser.add_argument("--input", type=str, required=True, help="Directory containing TSV files.")
    parser.add_argument("--output", type=str, required=True, help="Directory to write processed TSV files.")
    parser.add_argument("--index", type=str, required=True, help="Indexes to match, e.g. 1 or 1,2,3")

    args = parser.parse_args()

    index_list = [i.strip() for i in args.index.split(",")]

    process_tsv_folder(args.input, args.output, index_list)
