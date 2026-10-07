#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_boltz.sh <input_path> [out_dir] [cuda_device] [recycling_steps] [diffusion_samples]
# Example (invoking all default values: recycling_steps=10, diffusion_samples=25):
#   ./run_boltz.sh input/
# Specify every argument explicitly:
#   ./run_boltz.sh input/ ./out/ 0 10 25 (the default parameters for AlphaFold3)

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_path> [out_dir] [cuda_device] [recycling_steps] [diffusion_samples]" >&2
  exit 1
fi

# Required argument.
input_path="$1"

# Optional arguments and their default values.
out_dir="${2:-./boltz_output/}"
cuda_device="${3:-3}"
recycling_steps="${4:-3}"
diffusion_samples="${5:-1}"

# Create the output directory.
mkdir -p "$out_dir"

# Assign the GPU device.
export CUDA_VISIBLE_DEVICES="$cuda_device"

# Record the start time (Unix timestamp, in seconds).
start_ts=$(date +%s)

# Primary command (always executed in the foreground).
boltz predict "$input_path" \
    --out_dir "$out_dir" \
    --cache "/s0/dhuang/boltz/cache/" \
    --use_msa_server \
    --output_format pdb \
    --devices 1 \
    --recycling_steps "$recycling_steps" \
    --diffusion_samples "$diffusion_samples" \
  > grafting_boltz.log 2>&1

# Record the end time.
end_ts=$(date +%s)

# Compute the elapsed time and print it.
elapsed=$((end_ts - start_ts))
hours=$((elapsed/3600))
mins=$(((elapsed%3600)/60))
secs=$((elapsed%60))
printf "Done in %d:%02d:%02d (hh:mm:ss)\n" "$hours" "$mins" "$secs"

echo "Logs: grafting_boltz.log"
echo "Results: $out_dir"
