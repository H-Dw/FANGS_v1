#!/usr/bin/env bash
# AlphaFold 3 environment. Source this file in an interactive shell.
set -e
# Conda compiler activation/deactivation hooks may read unset backup variables.
# Disable nounset only during environment switching, then restore caller state.
_af3_restore_nounset=0
case $- in *u*) _af3_restore_nounset=1 ;; esac
set +u
_af3_activation_status=0
_af3_conda_exe="${CONDA_EXE:-}"
if [[ -z "$_af3_conda_exe" ]]; then
  _af3_conda_exe="$(type -P conda || true)"
fi
if [[ -z "$_af3_conda_exe" ]]; then
  echo "conda was not found. Activate a conda installation or set CONDA_EXE to the conda executable." >&2
  _af3_activation_status=1
else
  _af3_conda_root="$(cd -- "$(dirname -- "$_af3_conda_exe")/.." && pwd)"
  if [[ ! -f "$_af3_conda_root/etc/profile.d/conda.sh" ]]; then
    _af3_conda_root="$(cd -- "$_af3_conda_root/.." && pwd)"
  fi
  if [[ ! -f "$_af3_conda_root/etc/profile.d/conda.sh" ]]; then
    echo "Could not find conda.sh from $_af3_conda_exe." >&2
    _af3_activation_status=1
  else
    # shellcheck disable=SC1091
    source "$_af3_conda_root/etc/profile.d/conda.sh" || _af3_activation_status=$?
  fi
fi
if [[ $_af3_activation_status -eq 0 ]]; then
  conda activate "${AF3_CONDA_ENV:-alphafold3}" || _af3_activation_status=$?
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
if [[ -n "${AF3_ROOT:-}" ]]; then
  export AF3_ROOT
elif [[ -n "${AF3_HOME:-}" ]]; then
  export AF3_ROOT="${AF3_HOME}/alphafold3"
else
  echo "Set AF3_HOME to the directory that contains the alphafold3 checkout, or set AF3_ROOT to that checkout." >&2
  return 1 2>/dev/null || exit 1
fi
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
