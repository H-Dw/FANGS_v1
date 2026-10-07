#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_txt_file> <destination_directory>"
    exit 1
fi

input_file="$1"
destination_directory="$2"

if [ ! -d "$destination_directory" ]; then
    mkdir -p "$destination_directory"
fi

while IFS= read -r line
do
    cif_id="$line"
    # wget -P "$destination_directory" "https://files.rcsb.org/download/${cif_id}.cif"
    axel -n 10 -o "$destination_directory" "https://files.rcsb.org/download/${cif_id}.cif"
done < "$input_file"

echo "Download completed."