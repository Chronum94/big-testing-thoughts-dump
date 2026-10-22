"""Build the four intervention arms (identical size) + shared val/test.
  base   : all s1 teacher-MD frames (md/)
  scan   : base + rigid-molecule volume scans of every 3rd s1 frame at s = 0.95/1.05/1.10 (scan_s1/)
  aff    : base + affine (bond-stretching) scans of the same (frame, s) pairs (aff_s1/)
  dup    : base + one unchanged copy of the source frame per scan
  fresh  : base + new s1 teacher-MD frames (md_fresh/, seed 7, T + 50 K), trimmed/random-subsampled to the same count
Val: 48 random frames of the s2 teacher MD (md_s2/). Test: remaining s2 MD frames (test_md) + rigid scans of s2 (test_scan).
Seed 0. Writes data/<arm>/train.extxyz, data/val.extxyz, data/test_md.extxyz, data/test_scan.extxyz."""
import glob
import os

import numpy as np
from ase.io import read, write

rng = np.random.default_rng(0)
rd = lambda pat: [a for f in sorted(glob.glob(pat)) for a in read(f, ":", format="extxyz")]  # noqa: E731
base = rd("md/*.extxyz"); fresh = rd("md_fresh/*.extxyz"); s2 = rd("md_s2/*.extxyz")
key = lambda a: (a.info["src_frame"], round(a.info["scan_s"], 3))  # noqa: E731
scan = {key(a): a for a in rd("scan_s1/labelled_*.extxyz")}; aff = {key(a): a for a in rd("aff_s1/labelled_*.extxyz")}
common = sorted(set(scan) & set(aff)); print(f"rigid {len(scan)}, affine {len(aff)}, matched {len(common)}")
src = {}
for f in sorted(glob.glob("md/*.extxyz")):
    for k, a in enumerate(read(f, ":", format="extxyz")):
        src[f"{f}:{k}"] = a
n = len(common)
# dup uses the source Atoms objects themselves: Atoms.copy() drops the SinglePointCalculator (labels)
arms = {"scan": [scan[c] for c in common], "aff": [aff[c] for c in common], "dup": [src[c[0]] for c in common],
        "fresh": [fresh[i] for i in rng.choice(len(fresh), min(n, len(fresh)), replace=False)]}
for arm, extra in arms.items():
    os.makedirs(f"data/{arm}", exist_ok=True); write(f"data/{arm}/train.extxyz", base + extra)
    print(f"{arm}: {len(base)} + {len(extra)} = {len(base) + len(extra)}")
i = rng.permutation(len(s2)); write("data/val.extxyz", [s2[k] for k in i[:48]]); write("data/test_md.extxyz", [s2[k] for k in i[48:]])
write("data/test_scan.extxyz", rd("scan_s2/labelled_*.extxyz"))
print(f"val 48, test_md {len(s2) - 48}, test_scan {len(rd('scan_s2/labelled_*.extxyz'))}")
