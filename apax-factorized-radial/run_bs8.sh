#!/bin/bash
cd ~/scratch/factorized_radial
w=0
for m in dense r8j0.1 r64j0.1; do
  JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c $((w*5))-$((w*5+4)) \
    /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_mad5kbs8_${m}_s1.yaml > mad5kbs8_${m}_s1.out 2>&1 &
  w=$((w+1))
done
wait
for T in 5k 5kbs8; do echo "== $T"; TAG=$T PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_multi.py 1 dense r8j0.1 r64j0.1 2>/dev/null | sed 's/±nan//g'; done
/home/chronum/miniconda3/envs/apaxenv/bin/python -c "
import numpy as np
for m in ['dense','r8j0.1','r64j0.1']:
    t=np.genfromtxt(f'models/mad5kbs8_{m}_s1/log.csv',delimiter=',',names=True)['epoch_time']; print('bs8',m,'loop %.0f s, median %.2f s/ep'%(t.sum(),np.median(t[1:])))"
