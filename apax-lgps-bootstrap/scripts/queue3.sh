#!/usr/bin/env zsh
# Replaces queue.sh / queue_b.sh after lgB5_s2: final loss plot; EMA 0.9 run with the scanned epoch (epoch-scan branch);
# then the student MD (NVT 600/900 K 100 ps, NVE 600 K 20 ps, 400 atoms) for s1/s2; then the EMA 0.99 control.
cd /home/chronum/scratch/lgps
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
while [ -n "$(podman ps -q)" ]; do sleep 20; done
echo "$(date '+%F %T') done models/lgB5_s2 exit ? $(grep -c '' models/lgB5_s2/log.csv) lines" >> queue.log
~/apax/.venv/bin/python plot_loss.py lgB5_s1 lgB5_s2 && cp loss.png loss_s1_s2_final.png
train() {  # train <run> <apax src>
  echo "$(date '+%F %T') train models/$1 $1.yaml" >> queue.log
  APAX_SRC=$2 ~/scratch/jax-rocm-wsl/run-apax.sh $1.yaml > $1.out 2>&1
  echo "$(date '+%F %T') done models/$1 exit $? $(grep -c '' models/$1/log.csv 2>/dev/null) lines" >> queue.log
}
train lgB5_s1_ema90_scan /home/chronum/apax-wt/epoch-scan
export APAX_SRC=/home/chronum/apax-wt/mpi
for r in lgB5_s1 lgB5_s2; do
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
train lgB5_s1_ema99 /home/chronum/apax-wt/mpi
echo "$(date '+%F %T') queue done" >> queue.log
