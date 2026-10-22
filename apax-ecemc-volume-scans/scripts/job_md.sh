# EC:EMC/LiPF6 teacher MD: 21 boxes (7 compositions x 3 densities, seed 1), T cycles 300/400/500 K, 0.5 ps each (H mass 4, dt 2 fs). Resumable per run.
cd /home/chronum/scratch/ecemc; . /home/chronum/scratch/common/guard.sh
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
mkdir -p md; i=0
for comp in elec solv conc ec4 dmc lif hf; do for rho in 1.10 1.20 1.30; do
  T=$(( (i % 3) * 100 + 300 )); i=$((i + 1)); tag=${comp}_rho${rho}_s1; o=md/${tag}_T$T.extxyz
  [ -s $o ] && continue
  for try in 1 2 3; do Gw 10 4 $o.log /home/chronum/scratch/petjax/run_gpu.sh python -u teacher_md.py pack/$tag.extxyz $o $T 0.5 > $o.log 2>&1 && break; rm -f $o.tmp; done
done; done
