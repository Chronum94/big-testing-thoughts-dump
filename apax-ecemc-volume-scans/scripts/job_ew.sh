# energy-weighting arms: MACE-style end phase (from arm_gen), + pairwise, energy weight 100 from scratch, and a +50-epoch control
cd /home/chronum/scratch/ecemc; . /home/chronum/scratch/common/guard.sh
export APAX_SRC=/home/chronum/apax-wt/avgnbr XLA_FLAGS=--xla_gpu_enable_command_buffer=
for n in $(cat runs_ew.txt); do [ -f models/$n/DONE ] && continue
  for try in 1 2; do Gw 45 8 $n.out /home/chronum/scratch/jax-rocm-wsl/run-apax.sh $n.yaml > $n.out 2>&1 && grep -q "Finished training" $n.out && { touch models/$n/DONE; break; }; done
done
