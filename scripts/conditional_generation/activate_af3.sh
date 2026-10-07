#!/usr/bin/env bash
# AlphaFold 3 environment on ml-apus. Source this file in an interactive shell.
set -e
# Conda compiler activation/deactivation hooks may read unset backup variables.
# Disable nounset only during environment switching, then restore caller state.
_af3_restore_nounset=0
case $- in *u*) _af3_restore_nounset=1 ;; esac
set +u
_af3_activation_status=0
source /data1/dhuang/miniconda3/etc/profile.d/conda.sh || _af3_activation_status=$?
if [[ $_af3_activation_status -eq 0 ]]; then
  conda activate "${AF3_CONDA_ENV:-/data1/dhuang/miniconda3/envs/alphafold3}" || _af3_activation_status=$?
fi
if [[ $_af3_restore_nounset -eq 1 ]]; then
  set -u
fi
unset _af3_restore_nounset
if [[ $_af3_activation_status -ne 0 ]]; then
  printf 'AF3 Conda activation failed: %s\n' "$_af3_activation_status" >&2
  return "$_af3_activation_status" 2>/dev/null || exit "$_af3_activation_status"
fi
unset _af3_activation_status
export AF3_ROOT="${AF3_ROOT:-${AF3_HOME:-/data1/dhuang/af3}/alphafold3}"
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
# Upstream runtime tuning; these settings do not start a GPU job.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.95}"
# Set these in your shell only after locating existing weights/databases.
# export AF3_MODEL_DIR=/absolute/path/to/af3.bin.directory
# export AF3_DB_DIR=/absolute/path/to/existing/af3_databases
printf 'Activated AlphaFold 3: %s
' "$AF3_ROOT"
