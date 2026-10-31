#!/usr/bin/env bash
# full elastic tensors: LGPS (lowest Li ordering), PBA (K0/K4/K8), bcc Li; PET-MAD-m teacher + 2 students each (CPU); resumable, fails loudly
set -euo pipefail
cd /home/chronum/scratch/elastic
export JAX_PLATFORMS=cpu
S=/home/chronum/scratch
LG="LGPS=$S/lgps/relaxed_m.extxyz@0"
PB=("PBA_K0=$S/pba/relaxed_pba.extxyz@0" "PBA_K4=$S/pba/relaxed_pba.extxyz@3" "PBA_K8=$S/pba/relaxed_pba.extxyz@7")
LI="Li_bcc=$S/elastic/li_bcc.extxyz@0"
run() { taskset -c 0-7 "$@"; }
run $S/petjax/.venv/bin/python elastic_gen.py teacher $S/petjax/checkpoints/pet-mad-m sol_teacher.jsonl $LI $LG "${PB[@]}" >> logs/sol_teacher.out 2>&1
export PYTHONPATH=/home/chronum/apax-wt/avgnbr
for m in lgps/models/lgB5_s1 lgps/models/lgB5_s2; do run /home/chronum/miniconda3/envs/apaxenv/bin/python elastic_gen.py apax $S/$m sol_$(basename $m).jsonl $LG >> logs/sol_students.out 2>&1; done
for m in pba/models/pba_v0_s1 pba/models/pba_v0_asinh; do run /home/chronum/miniconda3/envs/apaxenv/bin/python elastic_gen.py apax $S/$m sol_$(basename $m).jsonl "${PB[@]}" >> logs/sol_students.out 2>&1; done
for m in li/models/li_v0 exp/gap23_md_splits/models/li_R5_s1; do run /home/chronum/miniconda3/envs/apaxenv/bin/python elastic_gen.py apax $S/$m sol_$(basename $m).jsonl $LI >> logs/sol_students.out 2>&1; done
[ "$(wc -l < sol_teacher.jsonl)" -eq 5 ] || { echo "teacher incomplete"; exit 1; }
