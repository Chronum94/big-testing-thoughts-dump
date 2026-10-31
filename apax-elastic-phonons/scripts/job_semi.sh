#!/usr/bin/env bash
# cubic semiconductor elastic constants: PET-MAD-m teacher + seed_v3/seed_v4 students (CPU); resumable, fails loudly
set -euo pipefail
cd /home/chronum/scratch/elastic
export JAX_PLATFORMS=cpu
taskset -c 0-7 /home/chronum/scratch/petjax/.venv/bin/python elastic.py teacher /home/chronum/scratch/petjax/checkpoints/pet-mad-m semi_teacher.jsonl > logs/semi_teacher.out 2>&1
for m in seed_v3 seed_v4; do
  PYTHONPATH=/home/chronum/apax-wt/shallow-uq taskset -c 0-7 /home/chronum/miniconda3/envs/apaxenv/bin/python elastic.py apax /home/chronum/scratch/semicond_dataset/models/$m semi_$m.jsonl > logs/semi_$m.out 2>&1
done
for f in semi_teacher semi_seed_v3 semi_seed_v4; do [ "$(wc -l < $f.jsonl)" -eq 27 ] || { echo "incomplete $f"; exit 1; }; done
