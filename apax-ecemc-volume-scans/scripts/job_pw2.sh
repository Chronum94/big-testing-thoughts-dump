# g20 x {r6t5, r5t5} x {pairwise 1e3, 0} and the g12w control (g12 strain, same wide source set as g20); runs_pw2.txt
cd /home/chronum/scratch/ecemc; . /home/chronum/scratch/common/guard.sh
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
P=/home/chronum/miniconda3/envs/apaxenv/bin/python
[ -f g12w_s1/LABELLED ] || { bash /home/chronum/scratch/common/label_by_size.sh g12w_s1 15 g12w_s1/unl.extxyz && touch g12w_s1/LABELLED || exit 1; }
[ -s data/g12w/train.extxyz ] || $P -c "
import glob, os; from ase.io import read, write
base = [a for f in sorted(glob.glob('md/*.extxyz')) for a in read(f, ':', format='extxyz')]
extra = [a for f in sorted(glob.glob('g12w_s1/labelled_*.extxyz')) for a in read(f, ':', format='extxyz')]
assert len(extra) == 399, len(extra)
os.makedirs('data/g12w', exist_ok=True); write('data/g12w/train.extxyz', base + extra); print('g12w', len(base), '+', len(extra))" || exit 1
export APAX_SRC=/home/chronum/apax-wt/avgnbr
for n in $(cat runs_pw2.txt); do [ -f models/$n/DONE ] && continue
  for try in 1 2; do Gw 45 8 $n.out /home/chronum/scratch/jax-rocm-wsl/run-apax.sh $n.yaml > $n.out 2>&1 && grep -q "Finished training" $n.out && { touch models/$n/DONE; break; }; done
done
for n in $(cat runs_pw2.txt); do [ -f models/$n/DONE ] || { echo "not trained: $n"; exit 1; }; done
