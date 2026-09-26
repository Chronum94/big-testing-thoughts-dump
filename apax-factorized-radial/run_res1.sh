#!/bin/bash
cd ~/scratch/factorized_radial
JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c 0-7 /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_mad2k5_r8j0.1res1_s1.yaml > mad2k5_r8j0.1res1_s1.out 2>&1
TAG=2k5 PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_multi.py 1 dense r8j0.1 r128j0.1 r8j0.1res10 r8j0.1res1 2>/dev/null | sed 's/±nan//g'
/home/chronum/miniconda3/envs/apaxenv/bin/python -c "
import numpy as np
L=np.genfromtxt('models/mad2k5_r8j0.1res1_s1/log.csv',delimiter=',',names=True); print('min train F %.3f, final train F %.3f, median s/ep %.2f @8c'%(L['train_forces_mae'].min(),L['train_forces_mae'][-1],np.median(L['epoch_time'][1:])))"
