#!/usr/bin/env bash
# Run AlphaFold 3 template-only inputs on four independent GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
AF3_HOME="${AF3_HOME:-/data1/dhuang/af3}"
AF3_REPO="${AF3_ROOT:-$AF3_HOME/alphafold3}"
CONFIG_ROOT=""
MODEL_DIR=""
OUTPUT_DIR=""
GPU_LIST="0,1,2,3"
NUM_DIFFUSION_SAMPLES=5
NUM_RECYCLES=10
FLASH_ATTENTION=triton
XLA_MEM_FRACTION=0.85
DISABLE_COMMAND_BUFFER=1
DRY_RUN=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: run_af3_4gpu.sh --config_root DIR --model_dir DIR --output_dir DIR [options]

Required:
  --config_root DIR    Generator output containing shards/gpu0 ... gpu3
  --model_dir DIR      Directory containing the licensed AF3 *.bin[.zst] weights
  --output_dir DIR     Prediction output root

Options:
  --gpus LIST          Four physical GPU IDs (default: 0,1,2,3)
  --num_diffusion_samples N  Default: 5
  --num_recycles N           Default: 10
  --flash_attention NAME     triton, cudnn, or xla (default: triton)
  --check_only               Check all GPU backends without prediction
  --dry_run                  Validate and print commands without using GPUs
  -h, --help                 Show this help

The generated JSON explicitly disables paired/unpaired MSA. This launcher also
sets --run_data_pipeline=false, so genetic databases and MSA files are not read.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_root) CONFIG_ROOT="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --num_diffusion_samples) NUM_DIFFUSION_SAMPLES="$2"; shift 2 ;;
    --num_recycles) NUM_RECYCLES="$2"; shift 2 ;;
    --flash_attention) FLASH_ATTENTION="$2"; shift 2 ;;
    --check_only) CHECK_ONLY=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$CONFIG_ROOT" && -n "$MODEL_DIR" && -n "$OUTPUT_DIR" ]] || { usage >&2; exit 2; }
[[ -d "$CONFIG_ROOT/shards" ]] || { echo "Missing $CONFIG_ROOT/shards" >&2; exit 2; }
[[ -d "$MODEL_DIR" ]] || { echo "Model directory does not exist: $MODEL_DIR" >&2; exit 2; }
[[ -f "$AF3_REPO/run_alphafold.py" ]] || { echo "Missing AF3 checkout: $AF3_REPO" >&2; exit 2; }
CONFIG_ROOT="$(realpath "$CONFIG_ROOT")"
MODEL_DIR="$(realpath "$MODEL_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[[ ${#GPUS[@]} -eq 4 ]] || { echo "--gpus must contain exactly four IDs" >&2; exit 2; }

# Isolate Python and CUDA dependencies from the caller's Conda/CUDA setup.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-$XLA_MEM_FRACTION}"
source "$SCRIPT_DIR/activate_af3.sh" >/dev/null
PYTHON="$CONDA_PREFIX/bin/python"
CUDA_LIB_DIRS=$("$PYTHON" - <<'PY'
from pathlib import Path
import sysconfig
root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
paths = sorted(root.glob("*/lib"))
if not list(root.glob("cusparse/lib/libcusparse.so*")):
    raise SystemExit("Missing AF3 environment cuSPARSE; restore locked CUDA dependencies.")
print(":".join(map(str, paths)))
PY
)
# Use only this environment's pip CUDA libraries and the system driver search
# paths. Do not inherit libraries from another environment or CUDA toolkit.
export LD_LIBRARY_PATH="$CUDA_LIB_DIRS"
export JAX_PLATFORMS=cuda
if [[ "$DISABLE_COMMAND_BUFFER" -eq 1 ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
fi
for worker in 0 1 2 3; do
  [[ -d "$CONFIG_ROOT/shards/gpu${worker}" ]] || { echo "Missing shard gpu${worker}" >&2; exit 2; }
done
if [[ "$DRY_RUN" -eq 0 ]]; then
  for gpu in "${GPUS[@]}"; do
    echo "Checking physical GPU $gpu with $PYTHON"
    CUDA_VISIBLE_DEVICES="$gpu" XLA_PYTHON_CLIENT_PREALLOCATE=false "$PYTHON" - <<'PY'
import ctypes
import os
import sys
sys.path.insert(0, os.environ["AF3_ROOT"])
ctypes.CDLL("libcusparse.so.12")
import run_alphafold
import jax
from jax_cuda12_plugin import _versions
print("cuSPARSE:", _versions.cusparse_get_version(), "JAX:", jax.__version__)
devices = jax.local_devices(backend="gpu")
if len(devices) != 1:
    raise SystemExit(f"Expected exactly one visible GPU, got {devices}")
print("GPU preflight OK:", devices)
PY
  done
fi
[[ "$CHECK_ONLY" -eq 1 ]] && exit 0
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/pids" "$OUTPUT_DIR/jax_cache"

pids=()
labels=()
for worker in 0 1 2 3; do
  shard="$CONFIG_ROOT/shards/gpu${worker}"
  [[ -d "$shard" ]] || { echo "Missing shard: $shard" >&2; exit 2; }
  count=$(find "$shard" -maxdepth 1 -type f -name '*.json' | wc -l)
  if [[ "$count" -eq 0 ]]; then
    echo "GPU worker $worker: empty shard; skipping"
    continue
  fi
  gpu="${GPUS[$worker]}"
  log="$OUTPUT_DIR/logs/gpu${gpu}.log"
  cache="$OUTPUT_DIR/jax_cache/gpu${gpu}"
  mkdir -p "$cache"
  cmd=("$PYTHON" "$AF3_REPO/run_alphafold.py"
    --input_dir="$shard"
    --output_dir="$OUTPUT_DIR/gpu${gpu}"
    --model_dir="$MODEL_DIR"
    --run_data_pipeline=false
    --run_inference=true
    --jax_backend=gpu
    --gpu_device=0
    --jax_compilation_cache_dir="$cache"
    --flash_attention_implementation="$FLASH_ATTENTION"
    --num_diffusion_samples="$NUM_DIFFUSION_SAMPLES"
    --num_recycles="$NUM_RECYCLES")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    echo "Starting worker $worker on physical GPU $gpu ($count configs); log: $log"
    (export CUDA_VISIBLE_DEVICES="$gpu"; "${cmd[@]}") >"$log" 2>&1 &
    pid=$!
    echo "$pid" > "$OUTPUT_DIR/pids/gpu${gpu}.pid"
    pids+=("$pid")
    labels+=("GPU $gpu")
  fi
done

[[ "$DRY_RUN" -eq 1 ]] && exit 0

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "${labels[$i]} completed successfully"
  else
    rc=$?
    echo "${labels[$i]} failed with exit code $rc" >&2
    status=1
  fi
done
exit "$status"
