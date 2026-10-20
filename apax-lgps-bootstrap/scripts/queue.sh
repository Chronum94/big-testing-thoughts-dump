#!/usr/bin/env zsh
# After boot.sh: split, train lgB5_s1/s2 (r 6, packed B5), then per model NVT 600 / 900 K (100 ps) and NVE 600 K (20 ps)
# on a 2x2x2 cell (400 atoms) of the lowest ordering.
cd /home/chronum/scratch/lgps
export APAX_SRC=/home/chronum/apax-wt/mpi XLA_FLAGS=--xla_gpu_enable_command_buffer=
until grep -q "boot done" md_teacher.out 2>/dev/null; do sleep 30; done
~/apax/.venv/bin/python split.py boot.extxyz >> queue.log 2>&1
~/apax/.venv/bin/python -c "from ase.io import read, write; write('lgps_start.extxyz', read('relaxed_all.extxyz', 0).repeat((2, 2, 2)))"
runs=(lgB5_s1 lgB5_s2)
for r in $runs; do
  echo "$(date '+%F %T') train models/$r $r.yaml" >> queue.log
  ~/scratch/jax-rocm-wsl/run-apax.sh $r.yaml > $r.out 2>&1
  echo "$(date '+%F %T') done models/$r exit $? $(grep -c '' models/$r/log.csv 2>/dev/null) lines" >> queue.log
done
for r in $runs; do
  for T in 600 900; do
    sed -e "s/MODEL/$r/" -e "s/TEMP/$T/g" md_nvt.tmpl > md/$r.nvt$T.yaml
    echo "$(date '+%F %T') md $r nvt$T" >> queue.log
    ./md.sh models/$r/config.yaml md/$r.nvt$T.yaml > md/$r.nvt$T.out 2>&1
    echo "$(date '+%F %T') md $r nvt$T exit $?" >> queue.log
  done
  sed "s/MODEL/$r/" md_nve.tmpl > md/$r.nve600.yaml
  ./md.sh models/$r/config.yaml md/$r.nve600.yaml > md/$r.nve600.out 2>&1
  echo "$(date '+%F %T') md $r nve600 exit $?" >> queue.log
done
echo "$(date '+%F %T') queue done" >> queue.log
