#!/usr/bin/env bash
# CPU-only apax container (no GPU device): ./run_cpu.sh script.py args...
exec podman run --rm -e JAX_PLATFORMS=cpu \
  -v /home/chronum/apax-wt/shallow-uq:/home/chronum/apax-wt/shallow-uq:ro -e PYTHONPATH=/home/chronum/apax-wt/shallow-uq \
  -v "$HOME/scratch":"$HOME/scratch" -w "$PWD" localhost/jax-rocm-wsl:apax python -u "$@"
