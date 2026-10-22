# anisotropic rigid scans, two arms: ortho (diagonal strain, orthorhombic cell) vs gen (symmetric strain incl. shear, triclinic).
# Same source frames / s values / volume change as the isotropic scan arm; traceless strain with Frobenius norm 0.06.
cd /home/chronum/scratch/ecemc; . /home/chronum/scratch/common/guard.sh
export APAX_SRC=/home/chronum/apax-wt/avgnbr XLA_FLAGS=--xla_gpu_enable_command_buffer=
P=/home/chronum/miniconda3/envs/apaxenv/bin/python
for arm in ortho gen; do mode=$arm; [ $arm = gen ] && mode=general; mkdir -p ${arm}_s1 ${arm}_s2
  [ -s ${arm}_s1/unl.extxyz ] || $P volume_scan.py ${arm}_s1/unl.extxyz md/*.extxyz --every 3 --scales 0.95,0.95,0.95,1.05,1.10 --aniso 0.06 --aniso_mode $mode --seed 1 || exit 1
  [ -s ${arm}_s2/unl.extxyz ] || $P volume_scan.py ${arm}_s2/unl.extxyz md_s2/*.extxyz --every 3 --scales 0.95,1.05,1.10 --aniso 0.06 --aniso_mode $mode --seed 2 || exit 1
done
for d in ortho_s1 ortho_s2 gen_s1 gen_s2; do [ -f $d/LABELLED ] || { bash /home/chronum/scratch/common/label_by_size.sh $d 15 $d/unl.extxyz && touch $d/LABELLED || exit 1; }; done
[ -s data/gen/train.extxyz ] || $P make_aniso_arms.py || exit 1
for arm in ortho gen; do [ -f models/arm_$arm/DONE ] && continue
  for try in 1 2 3; do Gw 40 6 arm_$arm.out /home/chronum/scratch/jax-rocm-wsl/run-apax.sh arm_$arm.yaml > arm_$arm.out 2>&1 && grep -q "Finished training" arm_$arm.out && { touch models/arm_$arm/DONE; break; }; done
done
