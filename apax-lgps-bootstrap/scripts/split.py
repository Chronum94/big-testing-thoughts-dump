"""Random 90/10 train/val split of the bootstrap set; test = teacher MD frames.

usage: python split.py boot.extxyz
"""
import glob
import sys

import numpy as np
from ase.io import read, write

f = read(sys.argv[1], ":")
idx = np.random.default_rng(0).permutation(len(f)); n_val = len(f) // 10
write("data/val.extxyz", [f[i] for i in idx[:n_val]])
write("data/train.extxyz", [f[i] for i in idx[n_val:]])
test = [a for p in sorted(glob.glob("test_md_*.extxyz")) for a in read(p, ":")]
write("data/test.extxyz", test)
print(f"train {len(f) - n_val}, val {n_val}, test {len(test)}")
