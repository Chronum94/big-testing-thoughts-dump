#!/usr/bin/env bash
# Teacher-label many extxyz files, one container per atom count (the teacher hangs on the first call after a shape change).
# usage: label_by_size.sh OUTDIR MINUTES_PER_SIZE in1.extxyz [in2 ...]   -> OUTDIR/labelled_<n>.extxyz
out=$1; m=$2; shift 2
. /home/chronum/scratch/common/guard.sh
~/miniconda3/envs/apaxenv/bin/python - "$out" "$@" <<'PY'
import sys, collections
from ase.io import read, write
g = collections.defaultdict(list)
for p in sys.argv[2:]:
    for a in read(p, ":"):
        g[len(a)].append(a)
for n, fr in g.items():
    write(f"{sys.argv[1]}/unlabelled_{n}.extxyz", fr); print(n, len(fr))
PY
for u in "$out"/unlabelled_*.extxyz; do
  n=$(basename "$u" .extxyz); n=${n#unlabelled_}
  for try in 1 2 3; do  # label_extxyz resumes; a host driver crash hangs the GPU, Gw kills it when the output stops growing
    Gw "$m" 8 "$out/labelled_$n.extxyz" /home/chronum/scratch/petjax/run_gpu.sh python -u /home/chronum/scratch/common/label_extxyz.py "$out/labelled_$n.extxyz" "$u" && break
    echo "label $n try $try failed"
  done
done
