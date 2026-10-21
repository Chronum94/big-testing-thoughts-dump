"""Start structures for MD: relaxed superionic-like bcc (rho 3.0) and fcc (rho 4.0) cells from the bootstrap.

usage: python pick_starts.py boot.extxyz out.extxyz   (writes [0] = bcc rho 3.0, [1] = fcc rho 4.0)
"""
import sys

from ase.io import read, write

f = read(sys.argv[1], ":")
pick = [next(a for a in f if a.info["category"] == c and abs(a.info["rho"] - r) < 1e-6) for c, r in (("S_bcc_relaxed", 3.0), ("S_fcc_relaxed", 4.0))]
for a in pick:
    a.calc = None
write(sys.argv[2], pick)
print([(a.info["category"], a.info["rho"], len(a)) for a in pick])
