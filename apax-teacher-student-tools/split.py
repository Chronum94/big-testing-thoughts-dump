"""Random train/val split of a bootstrap file (+ optional extra train files); test = given MD files.

usage: python split.py OUTDIR VAL_FRAC SEED boot.extxyz [--extra a.extxyz ...] [--test t1.extxyz ...]
"""
import argparse

import numpy as np
from ase.io import read, write

ap = argparse.ArgumentParser()
ap.add_argument("outdir"); ap.add_argument("val_frac", type=float); ap.add_argument("seed", type=int); ap.add_argument("boot")
ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--test", nargs="*", default=[])
a = ap.parse_args()
f = read(a.boot, ":") + [x for p in a.extra for x in read(p, ":")]
idx = np.random.default_rng(a.seed).permutation(len(f)); nv = int(len(f) * a.val_frac)
write(f"{a.outdir}/val.extxyz", [f[i] for i in idx[:nv]]); write(f"{a.outdir}/train.extxyz", [f[i] for i in idx[nv:]])
test = [x for p in a.test for x in read(p, ":")]
if test:
    write(f"{a.outdir}/test.extxyz", test)
print(f"{a.outdir}: train {len(f) - nv}, val {nv}, test {len(test)}")
