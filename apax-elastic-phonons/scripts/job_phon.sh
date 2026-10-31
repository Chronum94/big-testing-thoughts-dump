#!/usr/bin/env bash
# ndsc phonon bands, 27 cubic semiconductors: PET-MAD-m teacher + seed_v3/seed_v4 (CPU, each model's own relaxed a); resumable, fails loudly
set -euo pipefail
cd /home/chronum/scratch/phon
export JAX_PLATFORMS=cpu
E=/home/chronum/scratch/elastic
PYTHONPATH=/home/chronum/ase taskset -c 0-7 /home/chronum/scratch/petjax/.venv/bin/python phon.py teacher /home/chronum/scratch/petjax/checkpoints/pet-mad-m $E/semi_teacher.jsonl bands/teacher > logs/teacher.out 2>&1
for m in seed_v3 seed_v4; do
  PYTHONPATH=/home/chronum/ase:/home/chronum/apax-wt/shallow-uq taskset -c 0-7 /home/chronum/miniconda3/envs/apaxenv/bin/python phon.py apax /home/chronum/scratch/semicond_dataset/models/$m $E/semi_$m.jsonl bands/$m > logs/$m.out 2>&1
done
for d in teacher seed_v3 seed_v4; do [ "$(ls bands/$d/*.npz | wc -l)" -eq 27 ] || { echo "incomplete $d"; exit 1; }; done
