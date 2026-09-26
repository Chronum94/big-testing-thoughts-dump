#!/bin/bash
cd ~/scratch/factorized_radial
w=0
for m in r64j0.1 r8j0.1b12r7; do
  JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c $((w*4))-$((w*4+3)) \
    /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_mad20k_${m}_s1.yaml > mad20k_${m}_s1.out 2>&1 &
  w=$((w+1))
done
wait
TAG=20k PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_multi.py 1 r8j0.1 r64j0.1 r8j0.1b12r7 2>/dev/null | sed 's/±nan//g'
TAG=20k PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_fbins.py 1 r8j0.1 r64j0.1 r8j0.1b12r7 2>/dev/null
/home/chronum/miniconda3/envs/apaxenv/bin/python -c "
import numpy as np
for m in ['r64j0.1','r8j0.1b12r7']:
    t=np.genfromtxt(f'models/mad20k_{m}_s1/log.csv',delimiter=',',names=True)['epoch_time']; print(m,'loop %.0f s, median %.2f s/ep @4c'%(t.sum(),np.median(t[1:])))"
