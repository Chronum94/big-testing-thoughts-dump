#!/bin/zsh
# run benchmark lines from a file, two at a time, 1 core each
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/miniconda3/envs/devenv/bin/python
cat $1 | xargs -P 2 -L 1 sh -c 'tag=$0; '$PY' bench.py $tag "$@" > $tag.out 2>&1'
