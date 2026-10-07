#!/usr/bin/env bash
# Run the AF3 observed-chain CDR-template cohort without touching legacy outputs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
BASE="$PROJECT_ROOT/data/ESM3-Template_validation"
CONFIG_ROOT="$BASE/af3_template/af3_observed_chain_canonical_20260929"
OUTPUT_DIR="${AF3_OBSERVED_OUTPUT_DIR:-$BASE/af3_template/af3_observed_predictions_20260929}"

[[ -f "$CONFIG_ROOT/summary.json" ]] || {
  echo "Missing prepared observed-chain configurations: $CONFIG_ROOT" >&2
  exit 1
}

AF3_HOME="${AF3_HOME:-/data1/dhuang/af3}"
MODEL_DIR="${AF3_MODEL_DIR:-${AF3_ROOT:-$AF3_HOME/alphafold3}/checkpoints}"

exec bash "$SCRIPT_DIR/run_af3_4gpu.sh" \
  --config_root "$CONFIG_ROOT" \
  --model_dir "$MODEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --gpus 0,1,2,3 \
  --num_diffusion_samples 5 \
  --num_recycles 10 \
  --flash_attention triton \
  "$@"
