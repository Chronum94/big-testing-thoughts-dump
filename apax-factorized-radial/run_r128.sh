#!/bin/bash
cd ~/scratch/factorized_radial
while pgrep -f run_nn128.sh >/dev/null; do sleep 30; done
JAX_PLATFORMS=cpu PYTHONPATH=/home/chronum/apax taskset -c 0-7 /home/chronum/miniconda3/envs/apaxenv/bin/python run_cfg.py config_mad2k5_r128j0.1_s1.yaml > mad2k5_r128j0.1_s1.out 2>&1
TAG=2k5 PYTHONPATH=/home/chronum/apax JAX_PLATFORMS=cpu /home/chronum/miniconda3/envs/apaxenv/bin/python eval_multi.py 1 dense r8j0.1 r128j0.1 2>/dev/null | sed 's/±nan//g'
