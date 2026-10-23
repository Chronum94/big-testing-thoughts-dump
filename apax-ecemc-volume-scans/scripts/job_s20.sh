# strain norm 0.20 arms (o20 / g20): aniso scans (--every 2 to oversample past contact rejection), teacher labels, dose-matched arms, train
cd /home/chronum/scratch/ecemc; . /home/chronum/scratch/common/guard.sh
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
P=/home/chronum/miniconda3/envs/apaxenv/bin/python
mk() { d=$1; shift; mkdir -p $d; [ -s $d/unl.extxyz ] || $P volume_scan.py $d/unl.extxyz "$@" || exit 1; }
mk o20_s1 md/*.extxyz --every 2 --scales 0.95,0.95,0.95,1.05,1.10 --aniso 0.20 --aniso_mode ortho --seed 15
mk g20_s1 md/*.extxyz --every 2 --scales 0.95,0.95,0.95,1.05,1.10 --aniso 0.20 --aniso_mode general --seed 16
mk o20_s2 md_s2/*.extxyz --every 3 --scales 0.95,1.05,1.10 --aniso 0.20 --aniso_mode ortho --seed 25
mk g20_s2 md_s2/*.extxyz --every 3 --scales 0.95,1.05,1.10 --aniso 0.20 --aniso_mode general --seed 26
for d in o20_s2 g20_s2 o20_s1 g20_s1; do
  [ -f $d/LABELLED ] || { bash /home/chronum/scratch/common/label_by_size.sh $d 15 $d/unl.extxyz && touch $d/LABELLED || exit 1; }
done
[ -s data/g20/train.extxyz ] || $P make_strain_arms.py o20,g20 o20_s2,g20_s2 || exit 1
export APAX_SRC=/home/chronum/apax-wt/avgnbr
for n in $(cat runs_s20.txt); do [ -f models/$n/DONE ] && continue
  for try in 1 2; do Gw 45 8 $n.out /home/chronum/scratch/jax-rocm-wsl/run-apax.sh $n.yaml > $n.out 2>&1 && grep -q "Finished training" $n.out && { touch models/$n/DONE; break; }; done
done
for n in $(cat runs_s20.txt); do [ -f models/$n/DONE ] || { echo "not trained: $n"; exit 1; }; done
