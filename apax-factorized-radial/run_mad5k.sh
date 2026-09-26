#!/bin/bash
# 8 workers x 2 cores; job k goes to worker k % 8
cd ~/scratch/factorized_radial
jobs=(); for s in 1 2 3 4 5; do for v in dense r8 r16 r32; do jobs+=("${v}_s$s"); done; done
for w in 0 1 2 3 4 5 6 7; do
  ( for ((k=w; k<${#jobs[@]}; k+=8)); do
      j=${jobs[$k]}
      JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c $((w*2))-$((w*2+1)) \
        /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_mad5k_$j.yaml > mad5k_$j.out 2>&1
    done ) &
done
wait
echo ALL DONE
