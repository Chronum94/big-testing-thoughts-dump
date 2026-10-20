#!/usr/bin/env zsh
# lgB5_s1 isotropic NPT (1 bar, dt 2 fs, 400 atoms): 300/400/500 K 1 ns (frames every 200 fs),
# 600-900 K 200 ps (every 100 fs), 100 K 50 ps (every 100 fs).
cd /home/chronum/scratch/lgps
export APAX_SRC=/home/chronum/apax-wt/mpi XLA_FLAGS=--xla_gpu_enable_command_buffer=
r=lgB5_s1
for spec in "300 1000000 100" "400 1000000 100" "500 1000000 100" "600 200000 50" "700 200000 50" "800 200000 50" "900 200000 50" "100 50000 50"; do
  set -- ${=spec}; T=$1
  sed -e "s/MODEL/$r/" -e "s/TEMP/$T/g" -e "s/DUR/$2/" -e "s/SAMP/$3/" md_npt.tmpl > md/$r.npt$T.yaml
  echo "$(date '+%F %T') md $r npt$T" >> queue.log
  ./md.sh models/$r/config.yaml md/$r.npt$T.yaml > md/$r.npt$T.out 2>&1
  echo "$(date '+%F %T') md $r npt$T exit $?" >> queue.log
done
echo "$(date '+%F %T') queue5 done" >> queue.log
