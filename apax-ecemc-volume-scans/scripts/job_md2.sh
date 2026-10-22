# Teacher MD for the scan-vs-dup-vs-fresh intervention (0.5 ps, H mass 4, dt 2 fs; resumable per run):
#  md_s2/    second packings (s2) of all 21 composition x density boxes, same T cycle 300/400/500 K -> val/test only
#  md_fresh/ s1 packings again, new RNG seed 7, T cycle +50 K (350/450/550)                     -> the 'fresh' arm
cd /home/chronum/scratch/ecemc; . /home/chronum/scratch/common/guard.sh
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
mkdir -p md_s2 md_fresh
for set in s2 fresh; do i=0
  for comp in elec solv conc ec4 dmc lif hf; do for rho in 1.10 1.20 1.30; do
    if [ $set = s2 ]; then T=$(( (i % 3) * 100 + 300 )); tag=${comp}_rho${rho}_s2; seed=0; d=md_s2
    else T=$(( (i % 3) * 100 + 350 )); tag=${comp}_rho${rho}_s1; seed=7; d=md_fresh; fi
    i=$((i + 1)); o=$d/${tag}_T$T.extxyz; [ -s $o ] && continue
    for try in 1 2 3; do Gw 10 4 $o.log /home/chronum/scratch/petjax/run_gpu.sh python -u teacher_md.py pack/$tag.extxyz $o $T 0.5 $seed > $o.log 2>&1 && break; rm -f $o.tmp; done
  done; done
done
