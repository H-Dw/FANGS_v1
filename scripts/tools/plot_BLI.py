import argparse
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_file(file_path):
    """
    Parses a single file to extract concentration and data.
    """
    try:
        # 1. Read metadata (Concentration) from the 2nd line (index 1)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Ensure file has enough lines
            if len(lines) < 5:
                return None
            
            # Line 2: "Conc1\t{Value}"
            conc_line = lines[1].strip()
            if '\t' in conc_line:
                conc_str_val = conc_line.split('\t')[1]
                try:
                    # Convert Molar to Nanomolar (M * 1e9)
                    conc_val_m = float(conc_str_val)
                    conc_val_nm = conc_val_m * 1e9
                    # Format label for legend (e.g., "10.5 nM")
                    label = f"{conc_val_nm:g} nM" 
                except ValueError:
                    conc_val_nm = 0
                    label = "Unknown"
            else:
                conc_val_nm = 0
                label = "Unknown"

        # 2. Read data: Header is the 5th line (index 4), so we skip the first 4 lines
        # treated as TSV regardless of extension
        df = pd.read_csv(file_path, sep='\t', skiprows=4)
        
        # Check if required columns exist
        if 'Time1' in df.columns and 'Data1' in df.columns:
            # Keep only relevant data and add the label column for plotting
            df_subset = df[['Time1', 'Data1']].copy()
            df_subset['Condition'] = label
            df_subset['Conc_Value'] = conc_val_nm # Used for sorting
            return df_subset
        else:
            print(f"Warning: Columns Time1/Data1 not found in {file_path}")
            return None

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # --- CONFIGURATION ---
    parser = argparse.ArgumentParser(description="Plot BLI result")
    parser.add_argument("--input", required=True, help="Input file or folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--font_size", default=20, type=int, help="Font size for plots")
    args = parser.parse_args()
    
    input_root_folder = args.input   
    output_folder = args.output
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Publication font sizes. Seaborn's set_theme() overwrites rcParams, so
    # apply sizes after the theme, then set each artist explicitly as well.
    FONTSIZE = args.font_size
    FONTWEIGHT = "bold"

    # Set Seaborn theme for SCI style
    # 'muted' palette provides low saturation colors suitable for publication
    sns.set_theme(style="ticks", palette="muted")
    plt.rcParams.update({
        "font.size": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "axes.labelweight": FONTWEIGHT,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "legend.frameon": False,
    }) 
    
    # Get all subdirectories in the input folder
    subdirs = [d for d in glob.glob(os.path.join(input_root_folder, '*')) if os.path.isdir(d)]

    if not subdirs:
        print("No subdirectories found inside the input folder.")
        # If the input itself is the target folder (contains files directly), handle that:
        if len(glob.glob(os.path.join(input_root_folder, '*'))) > 0:
            print("Checking if input folder contains files directly...")
            subdirs = [input_root_folder]
        else:
            return

    for subdir in subdirs:
        subdir_name = os.path.basename(os.path.normpath(subdir))
        print(f"Processing folder: {subdir_name}...")
        
        # Get all files in the subdirectory
        files = glob.glob(os.path.join(subdir, '*'))
        
        all_data = []

        for file_path in files:
            # Skip system files like .DS_Store or directories
            if os.path.basename(file_path).startswith('.') or os.path.isdir(file_path):
                continue
                
            df_file = parse_file(file_path)
            if df_file is not None:
                all_data.append(df_file)
        
        if not all_data:
            print(f"  No valid data found in {subdir_name}, skipping.")
            continue

        # Combine all dataframes for this folder
        combined_df = pd.concat(all_data, ignore_index=True)

        # Sort by Concentration Value (Low -> High) so the legend matches the visual stack
        combined_df.sort_values(by='Conc_Value', ascending=False, inplace=True)

        # --- PLOTTING ---
        plt.figure(figsize=(8, 6)) # Size in inches
        
        # Draw Line Plot
        ax = sns.lineplot(
            data=combined_df, 
            x='Time1', 
            y='Data1', 
            hue='Condition',
            linewidth=1.5
        )

        # Rename Axes
        ax.set_xlabel("Time (s)", fontsize=FONTSIZE, fontweight=FONTWEIGHT)
        ax.set_ylabel("Response (nm)", fontsize=FONTSIZE, fontweight=FONTWEIGHT)
        ax.tick_params(axis="both", labelsize=FONTSIZE)

        # Customize X-axis ticks (Every 200 units)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))

        # Scientific styling: Remove Top and Right spines (borders)
        sns.despine(top=True, right=True)

        # --- LEGEND CONFIGURATION (Right Side Upper) ---
        # bbox_to_anchor=(1.02, 1): Places the anchor point just outside the axes at the top-right.
        # loc='upper left': Aligns the top-left corner of the legend box to that anchor point.
        # This ensures the legend is on the right side, top aligned, without covering data.
        # Legend does not accept fontweight=; use prop (FontProperties) for size + weight.
        legend = ax.legend(
            title=None,
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            prop={"size": FONTSIZE, "weight": FONTWEIGHT},
        )
        legend.set_title(None)

        # Save plot
        output_filename = f"{subdir_name}.tif"
        save_path = os.path.join(output_folder, output_filename)
        
        # Save as TIF, 600 DPI, tight layout (crucial for outside legends)
        plt.savefig(save_path, dpi=600, format='tif', bbox_inches='tight')
        
        # Close the plot to free memory
        plt.close()
        
        print(f"  Saved: {save_path}")

    print("Processing complete.")

if __name__ == "__main__":
    main()