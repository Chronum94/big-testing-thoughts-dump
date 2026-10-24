# NVE 10 ps (dt 0.5 fs, 7722 atoms, 300 K start) per radial variant on arm_g20_r6t5_l1e3_s1; apax branch radial-tabulation
cd /home/chronum/scratch/ecemc/md_radial
export APAX_SRC=/home/chronum/apax-wt/radial-tab XLA_FLAGS=--xla_gpu_enable_command_buffer=
for v in exact pair_table hermite256 piecewise16_p7; do
  [ -s $v/md.h5 ] && grep -q "simulation finished\|Simulation finished\|100%" $v.out 2>/dev/null && continue
  /home/chronum/scratch/common/md.sh model_g20.yaml md_$v.yaml > $v.out 2>&1
  [ -s $v/md.h5 ] || { echo "no trajectory for $v"; tail -20 $v.out; exit 1; }
done
