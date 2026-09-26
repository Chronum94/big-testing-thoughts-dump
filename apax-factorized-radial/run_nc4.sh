#!/bin/bash
cd ~/scratch/factorized_radial
mapfile -t jobs < nc4_jobs.txt
for w in 0 1 2 3 4 5 6 7; do
  ( for ((k=w; k<${#jobs[@]}; k+=8)); do
      JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c $((w*2))-$((w*2+1)) \
        /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_${jobs[$k]}.yaml > ${jobs[$k]}.out 2>&1
    done ) &
done
wait
for T in 2k5 5k; do echo "== $T"; TAG=$T PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_multi.py 1,2,3 dense densenc4 r8j0.1 r8j0.1nc4 2>/dev/null; done
