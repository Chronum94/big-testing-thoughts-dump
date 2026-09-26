#!/bin/bash
# waits for the n_contr grid, then runs u-only / v-only (5k, rank 8, jitter 0.1, n_contr 8, 3 seeds)
cd ~/scratch/factorized_radial
while pgrep -f run_nc4.sh >/dev/null; do sleep 30; done
mapfile -t jobs < uv_jobs.txt
w=0
for j in "${jobs[@]}"; do
  JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c $((w*2))-$((w*2+1)) \
    /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_$j.yaml > $j.out 2>&1 &
  w=$((w+1))
done
wait
TAG=5k PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_multi.py 1,2,3 dense r8j0.1 r8j0.1centre r8j0.1nbr 2>/dev/null
