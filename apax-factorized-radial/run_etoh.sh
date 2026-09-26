#!/bin/bash
cd ~/scratch/factorized_radial
mapfile -t jobs < etoh_jobs.txt
w=0
for j in "${jobs[@]}"; do
  JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c $((w*3))-$((w*3+2)) \
    /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_$j.yaml > $j.out 2>&1 &
  w=$((w+1))
done
wait
/home/chronum/miniconda3/envs/apaxenv/bin/python -c "
import numpy as np
runs={'dense':['ab_fr_base','ab_fr_dense_s2','ab_fr_dense_s3'],'r8j0.1':[f'ab_fr_r8j0.1_s{s}' for s in (1,2,3)]}
for m,ds in runs.items():
    r=np.array([[np.genfromtxt(f'models/{d}/log.csv',delimiter=',',names=True)[-1][k]*1e3 for k in ('val_energy_mae','val_forces_mae','train_forces_mae')] for d in ds])
    print(m.ljust(7),'valE %.1f±%.1f meV  valF %.1f±%.1f meV/A  trainF %.1f±%.1f'%(r[:,0].mean(),r[:,0].std(ddof=1),r[:,1].mean(),r[:,1].std(ddof=1),r[:,2].mean(),r[:,2].std(ddof=1)), ' per-seed F:', np.round(r[:,1],1))"
