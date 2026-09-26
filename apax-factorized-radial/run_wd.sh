#!/bin/bash
# 4 slots on cores 8-15; slot k runs jobs k, k+4
cd ~/scratch/factorized_radial
jobs=(1e-5 1e-4 1e-3 1e-2 1e-1)
for w in 0 1 2 3; do
  ( for ((k=w; k<${#jobs[@]}; k+=4)); do
      JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c $((8+w*2))-$((9+w*2)) \
        /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_mad5k_dense_wd${jobs[$k]}_s1.yaml > mad5k_dense_wd${jobs[$k]}_s1.out 2>&1
    done ) &
done
wait; echo WD DONE
