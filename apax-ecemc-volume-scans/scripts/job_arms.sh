# scan-vs-affine-vs-dup-vs-fresh intervention: build scans, teacher-label them, build arms, train 4 students (200 epochs each)
cd /home/chronum/scratch/ecemc; . /home/chronum/scratch/common/guard.sh
export APAX_SRC=/home/chronum/apax-wt/avgnbr XLA_FLAGS=--xla_gpu_enable_command_buffer=
P=/home/chronum/miniconda3/envs/apaxenv/bin/python
mkdir -p scan_s1 aff_s1 scan_s2
[ -s scan_s1/unl.extxyz ] || $P volume_scan.py scan_s1/unl.extxyz md/*.extxyz --every 3 --scales 0.95,1.05,1.10 || exit 1
[ -s aff_s1/unl.extxyz ] || $P volume_scan.py aff_s1/unl.extxyz md/*.extxyz --every 3 --scales 0.95,1.05,1.10 --affine || exit 1
[ -s scan_s2/unl.extxyz ] || $P volume_scan.py scan_s2/unl.extxyz md_s2/*.extxyz --every 3 --scales 0.95,1.05,1.10 || exit 1
for d in scan_s1 aff_s1 scan_s2; do [ -f $d/LABELLED ] || { bash /home/chronum/scratch/common/label_by_size.sh $d 15 $d/unl.extxyz && touch $d/LABELLED || exit 1; }; done
[ -s data/scan/train.extxyz ] || $P make_arms.py || exit 1
for arm in scan aff dup fresh; do [ -f models/arm_$arm/DONE ] && continue
  for try in 1 2 3; do Gw 40 6 arm_$arm.out /home/chronum/scratch/jax-rocm-wsl/run-apax.sh arm_$arm.yaml > arm_$arm.out 2>&1 && grep -q "Finished training" arm_$arm.out && { touch models/arm_$arm/DONE; break; }; done
done
