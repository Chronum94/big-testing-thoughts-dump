#!/usr/bin/env bash
# apax jax-md in the ROCm container: APAX_SRC=... ./md.sh <model config.yaml> <md config.yaml>
HERE=$HOME/scratch/jax-rocm-wsl
APAX_SRC=$(realpath "${APAX_SRC:-$HOME/apax}")
DXG=$HERE/rocdxg-1.2.2/root/opt/rocm
exec podman run --rm \
  --device /dev/dxg --security-opt seccomp=unconfined --ipc=host \
  -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so:ro \
  -v "$DXG/lib/librocdxg.so.1.2.2":/usr/lib/librocdxg.so:ro \
  -v "$DXG/share/rocdxg/dids.conf":/usr/share/rocdxg/dids.conf:ro \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75} -e XLA_FLAGS="$XLA_FLAGS" \
  -e JAX_COMPILATION_CACHE_DIR=$HERE/jaxcache -v $HERE/jaxcache:$HERE/jaxcache \
  -v "$APAX_SRC":"$APAX_SRC":ro -e PYTHONPATH="$APAX_SRC" \
  -v "$HOME/scratch":"$HOME/scratch" -w /home/chronum/scratch/lgps \
  localhost/jax-rocm-wsl:apax \
  python -u -c "import jax; jax.devices()
import tensorflow  # must load after ROCm backend init but before any JAX compile, or the two LLVMs clash
import rocm_argsort_patch  # int64 argsort is broken on ROCm; keeps jax-md cell lists correct
import sys; from apax.md.simulate import run_md; run_md(sys.argv[1], sys.argv[2], log_level='info')" "$1" "$2"
