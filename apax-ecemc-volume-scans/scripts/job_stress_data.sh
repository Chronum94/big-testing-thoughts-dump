# Stress-test data: distribution-shift teacher MD, anisotropic scans at other strain norms, shear / uniaxial / swap test sets, labels.
cd /home/chronum/scratch/ecemc; . /home/chronum/scratch/common/guard.sh
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
P=/home/chronum/miniconda3/envs/apaxenv/bin/python; mkdir -p md_shift
# 1. distribution-shift MD (0.5 ps, H mass 4, dt 2 fs): new packings at 1.0 / 1.4 g/cm3 at 300 and 600 K; 2x electrolyte boxes at 300 / 450 K
for tag in elec solv conc dmc; do for rho in 1.00 1.40; do for T in 300 600; do o=md_shift/${tag}_rho${rho}_s3_T$T.extxyz; [ -s $o ] && continue
  for try in 1 2 3; do Gw 10 4 $o.log /home/chronum/scratch/petjax/run_gpu.sh python -u teacher_md.py pack/${tag}_rho${rho}_s3.extxyz $o $T 0.5 3 > $o.log 2>&1 && break; rm -f $o.tmp; done
done; done; done
for s in 3 4; do T=$(( s == 3 ? 300 : 450 )); o=md_shift/elecx2_rho1.20_s${s}_T$T.extxyz; [ -s $o ] && continue
  for try in 1 2 3; do Gw 15 5 $o.log /home/chronum/scratch/petjax/run_gpu.sh python -u teacher_md.py pack/elecx2_rho1.20_s$s.extxyz $o $T 0.5 3 > $o.log 2>&1 && break; rm -f $o.tmp; done
done
# 2. anisotropic scans (CPU): training sets at norms 0.03 / 0.12; held-out s2 test sets
mk() { d=$1; shift; mkdir -p $d; [ -s $d/unl.extxyz ] || $P volume_scan.py $d/unl.extxyz "$@" || exit 1; }
mk o03_s1 md/*.extxyz --every 3 --scales 0.95,0.95,0.95,1.05,1.10 --aniso 0.03 --aniso_mode ortho --seed 11
mk g03_s1 md/*.extxyz --every 3 --scales 0.95,0.95,0.95,1.05,1.10 --aniso 0.03 --aniso_mode general --seed 12
mk o12_s1 md/*.extxyz --every 3 --scales 0.95,0.95,0.95,1.05,1.10 --aniso 0.12 --aniso_mode ortho --seed 13
mk g12_s1 md/*.extxyz --every 3 --scales 0.95,0.95,0.95,1.05,1.10 --aniso 0.12 --aniso_mode general --seed 14
mk o12_s2 md_s2/*.extxyz --every 3 --scales 0.95,1.05,1.10 --aniso 0.12 --aniso_mode ortho --seed 21
mk g12_s2 md_s2/*.extxyz --every 3 --scales 0.95,1.05,1.10 --aniso 0.12 --aniso_mode general --seed 22
mk shear_s2 md_s2/*.extxyz --every 3 --scales 1.0,1.0 --aniso 0.08 --aniso_mode general --seed 23
mk uniax_s2 md_s2/*.extxyz --every 3 --scales 1.0,1.0 --aniso 0.15 --aniso_mode ortho --seed 24
# 3. teacher labels (one container per atom count; resumable)
for d in swap_emc swap_all shear_s2 uniax_s2 o12_s2 g12_s2 o03_s1 g03_s1 o12_s1 g12_s1; do
  [ -f $d/LABELLED ] || { bash /home/chronum/scratch/common/label_by_size.sh $d 15 $d/unl.extxyz && touch $d/LABELLED || exit 1; }
done
# 4. datasets
$P make_strain_arms.py o03,g03,o12,g12 o12_s2,g12_s2,shear_s2,uniax_s2,swap_all,swap_emc || exit 1
$P -c "
from ase.io import read, write; import glob
fr = [a for f in sorted(glob.glob('md_shift/*_s3_T*.extxyz')) if 'x2' not in f for a in read(f, ':', format='extxyz')]; write('data/test_shift.extxyz', fr)
x2 = [a for f in sorted(glob.glob('md_shift/elecx2*.extxyz')) for a in read(f, ':', format='extxyz')]; write('data/test_x2.extxyz', x2); print('shift', len(fr), 'x2', len(x2))"
touch STRESS_DATA_DONE
