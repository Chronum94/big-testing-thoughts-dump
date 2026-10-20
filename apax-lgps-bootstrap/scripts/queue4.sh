#!/usr/bin/env zsh
# lgB5_s1 NVT at dt 2 fs (100 ps; frames every 50 fs): 600/700/800/900 K, then NVE 600 K 20 ps at dt 2 fs.
cd /home/chronum/scratch/lgps
export APAX_SRC=/home/chronum/apax-wt/mpi XLA_FLAGS=--xla_gpu_enable_command_buffer=
r=lgB5_s1
for T in 700 800 600 900; do
  sed -e "s/MODEL/$r/" -e "s/TEMP/$T/g" md_nvt_dt2.tmpl > md/$r.dt2_nvt$T.yaml
  echo "$(date '+%F %T') md $r dt2_nvt$T" >> queue.log
  ./md.sh models/$r/config.yaml md/$r.dt2_nvt$T.yaml > md/$r.dt2_nvt$T.out 2>&1
  echo "$(date '+%F %T') md $r dt2_nvt$T exit $?" >> queue.log
done
sed "s/MODEL/$r/" md_nve_dt2.tmpl > md/$r.dt2_nve600.yaml
./md.sh models/$r/config.yaml md/$r.dt2_nve600.yaml > md/$r.dt2_nve600.out 2>&1
echo "$(date '+%F %T') md $r dt2_nve600 exit $?" >> queue.log
echo "$(date '+%F %T') queue4 done" >> queue.log
