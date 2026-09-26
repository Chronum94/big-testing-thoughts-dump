#!/bin/bash
cd ~/scratch/factorized_radial
JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c 0-7 /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_mad20k_cmnnbessel_s1.yaml > mad20k_cmnnbessel_s1.out 2>&1
TAG=20k PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_multi.py 1 r8j0.1 r64j0.1 r8j0.1b12r7 r8j0.1nn128 cmnnbessel 2>/dev/null | sed 's/±nan//g'
TAG=20k PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_fbins.py 1 r8j0.1 cmnnbessel 2>/dev/null
/home/chronum/miniconda3/envs/apaxenv/bin/python -c "
import numpy as np
L=np.genfromtxt('models/mad20k_cmnnbessel_s1/log.csv',delimiter=',',names=True); print('min train F %.3f, final train F %.3f, median s/ep %.2f @8c'%(L['train_forces_mae'].min(),L['train_forces_mae'][-1],np.median(L['epoch_time'][1:])))"
