#!/usr/bin/env bash
# GPU cost map: ./run_costmap.sh out.jsonl ['[[A,B,[drops]],...]']  (same container setup as run-apax.sh)
R=/home/chronum/scratch/jax-rocm-wsl; DXG=$R/rocdxg-1.2.2/root/opt/rocm
APAX_SRC=$(realpath "${APAX_SRC:-$HOME/apax-wt/shallow-uq}")
cd "$(dirname "$0")"
exec podman run --rm --device /dev/dxg --security-opt seccomp=unconfined --ipc=host \
  -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so:ro \
  -v "$DXG/lib/librocdxg.so.1.2.2":/usr/lib/librocdxg.so:ro \
  -v "$DXG/share/rocdxg/dids.conf":/usr/share/rocdxg/dids.conf:ro \
  -e HSA_ENABLE_DXG_DETECTION=1 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  -e JAX_COMPILATION_CACHE_DIR=$R/jaxcache -v $R/jaxcache:$R/jaxcache \
  -v "$APAX_SRC":"$APAX_SRC":ro -e PYTHONPATH="$APAX_SRC" -v "$HOME/scratch":"$HOME/scratch" -w "$PWD" \
  localhost/jax-rocm-wsl:apax python -u costmap.py "$@"
