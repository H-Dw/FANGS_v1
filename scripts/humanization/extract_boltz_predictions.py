import os
import shutil
import sys

def copy_files(base_dir, output_dir):
    """
    Extract the designated files from every immediate subdirectory of base_dir
    and write them separately into the pdb and json subdirectories of output_dir.
    """

    # Verify that the base directory exists.
    if not os.path.exists(base_dir):
        raise RuntimeError(f"[Error] Base directory does not exist: {base_dir}")
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"[Error] Base path is not a directory: {base_dir}")

    # Create the output directory structure.
    pdb_dir = os.path.join(output_dir, "pdb")
    json_dir = os.path.join(output_dir, "json")
    try:
        os.makedirs(pdb_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"[Error] Failed to create the output directory: {e}")

    valid = False  # Indicates whether at least one file has been copied.

    # Iterate over immediate subdirectories.
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            subfolder_name = entry.name
            subfolder_path = entry.path

            # Construct the source file paths.
            pdb_src = os.path.join(subfolder_path, f"{subfolder_name}_model_0.pdb")
            json_src = os.path.join(subfolder_path, f"confidence_{subfolder_name}_model_0.json")

            # Construct the destination file paths.
            # pdb_dst = os.path.join(pdb_dir, f"{subfolder_name}_model_0.pdb")
            # json_dst = os.path.join(json_dir, f"confidence_{subfolder_name}_model_0.json")
            pdb_dst = os.path.join(pdb_dir, f"{subfolder_name}.pdb")
            json_dst = os.path.join(json_dir, f"confidence_{subfolder_name}.json")

            # Copy the PDB file.
            if os.path.exists(pdb_src):
                try:
                    shutil.copy2(pdb_src, pdb_dst)
                    valid = True
                except Exception as e:
                    print(f"[Error] Failed to copy the PDB file: {pdb_src} -> {pdb_dst}. Reason: {e}")
            else:
                print(f"[Warning] PDB file not found; skipping: {pdb_src}")

            # Copy the JSON file.
            if os.path.exists(json_src):
                try:
                    shutil.copy2(json_src, json_dst)
                    valid = True
                except Exception as e:
                    print(f"[Error] Failed to copy the JSON file: {json_src} -> {json_dst}. Reason: {e}")
            else:
                print(f"[Warning] JSON file not found; skipping: {json_src}")

    # Verify that at least one file was copied.
    if not valid:
        raise RuntimeError("[Error] No valid files were copied. Exiting.")

    print("[Complete] All eligible files were copied successfully.")

# Program entry point.
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_boltz_predictions.py <base_directory> <output_directory>")
        sys.exit(1)

    base_dir = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        copy_files(base_dir, output_dir)
    except RuntimeError as e:
        print(e)
        sys.exit(1)