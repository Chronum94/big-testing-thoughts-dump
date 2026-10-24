# Evaluate one model on every stress-test set that exists; one log per (model, set) in logs_eval/ (skips existing).
# usage: bash eval_model.sh arm_xxx    (run inside pueue group cpu: cores picked from PUEUE_WORKER_ID, 4 per worker)
cd /home/chronum/scratch/ecemc; n=$1; a=${n#arm_}; w=${PUEUE_WORKER_ID:-0}; c0=$(( (w % 3) * 4 )); cores=$c0-$((c0 + 3))
export PYTHONPATH=${APAX_EVAL_SRC:-/home/chronum/apax-wt/avgnbr} JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
P=(taskset -c $cores /home/chronum/miniconda3/envs/apaxenv/bin/python); mkdir -p logs_eval
for t in md gen scan ortho shear uniax o12 g12 o20 g20 shift x2; do f=logs_eval/${a}_$t.txt; [ -s data/test_$t.extxyz ] || continue
  grep -q "^| all" $f 2>/dev/null || "${P[@]}" eval_inter_intra.py models/$n data/test_$t.extxyz 2>&1 | grep -v -i warn > $f
  grep -q "^| all" $f || { echo "no result table: $f"; head -5 $f; exit 1; }
done
f=logs_eval/${a}_pairs.txt; grep -q scanfam $f 2>/dev/null || "${P[@]}" eval_pairs.py models/$n 2>&1 | grep -v -i warn > $f
grep -q scanfam $f || { echo "no result: $f"; head -5 $f; exit 1; }
