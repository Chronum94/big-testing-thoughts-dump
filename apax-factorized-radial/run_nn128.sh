#!/bin/bash
cd ~/scratch/factorized_radial
JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c 0-7 /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_mad20k_r8j0.1nn128_s1.yaml > mad20k_r8j0.1nn128_s1.out 2>&1
TAG=20k PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_multi.py 1 r8j0.1 r8j0.1nn128 2>/dev/null | sed 's/±nan//g'
TAG=20k PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_fbins.py 1 r8j0.1 r8j0.1nn128 2>/dev/null
/home/chronum/miniconda3/envs/apaxenv/bin/python -c "
import numpy as np; t=np.genfromtxt('models/mad20k_r8j0.1nn128_s1/log.csv',delimiter=',',names=True)['epoch_time']; print('loop %.0f s, median %.2f s/ep @8c'%(t.sum(),np.median(t[1:])))"
