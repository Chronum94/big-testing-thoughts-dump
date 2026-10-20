#!/usr/bin/env zsh
# GPU: bootstrap set, then teacher NVT test trajectories (orderings 0 = lowest and the last = Ge (2,3)), 5 ps each.
cd /home/chronum/scratch/lgps
R=~/scratch/petjax/run_gpu.sh
$R python -u gen_boot.py boot.extxyz > gen_boot.out 2>&1
for i in 0 -1; do for T in 600 900; do
  $R python -u md_teacher.py relaxed_all.extxyz $i $T 5 test_md_${i}_${T}.extxyz >> md_teacher.out 2>&1
done; done
echo "boot done" >> md_teacher.out
