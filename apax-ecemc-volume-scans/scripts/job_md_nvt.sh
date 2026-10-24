# small box (151 atoms) NVT 200 ps at 330 K, dt 1 fs, H mass 4: exact (seeds 1, 2), pair_table, hermite256
cd /home/chronum/scratch/ecemc/md_small
export APAX_SRC=/home/chronum/apax-wt/radial-tab XLA_FLAGS=--xla_gpu_enable_command_buffer=
for v in exact_s1 exact_s2 pair_table_s1 hermite256_s1; do
  grep -q "simulation finished" nvt_$v.out 2>/dev/null && continue
  /home/chronum/scratch/common/md.sh model_g20.yaml md_nvt_$v.yaml > nvt_$v.out 2>&1
  grep -q "simulation finished" nvt_$v.out || { echo "run $v did not finish"; tail -20 nvt_$v.out; exit 1; }
done
