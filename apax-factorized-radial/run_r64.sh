#!/bin/bash
cd ~/scratch/factorized_radial
JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c 0-7 /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_mad5k_r64j0.1_s1.yaml > mad5k_r64j0.1_s1.out 2>&1
PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_multi.py 1 dense r8j0.1 r64j0.1 2>/dev/null | sed 's/±nan//g'
/home/chronum/miniconda3/envs/apaxenv/bin/python -c "
import numpy as np
for m in ['r8j0.1','r64j0.1']:
    L=np.genfromtxt(f'models/mad5k_{m}_s1/log.csv',delimiter=',',names=True)
    print(m, 'log final train F %.3f val F %.3f, median s/epoch %.2f'%(L['train_forces_mae'][-1],L['val_forces_mae'][-1],np.median(L['epoch_time'][1:])))"
