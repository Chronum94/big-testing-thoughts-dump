"""Val force error binned by |F_ref| per atom (norm error), plus median. usage: TAG=10k eval_fbins.py seed models..."""
import os, sys
import numpy as np
from ase.io import read
from apax.md import ASECalculator

TAG = os.environ.get("TAG", "10k")
A = read({"5k": "mad_dimtri_5000.traj", "10k": "mad_dimtri_10000.traj", "20k": "mad_dimtri_20000.traj"}[TAG], ":")
s, models = sys.argv[1], sys.argv[2:]
bins = [0, 1, 5, 20, 50, np.inf]
print(f"{'model':9s} | median |dF| | " + " | ".join(f"|F| {lo}-{hi}" for lo, hi in zip(bins[:-1], bins[1:])))
for m in models:
    d = f"models/mad{TAG}_{m}_s{s}"
    va = [A[i] for i in np.load(f"{d}/train_val_idxs.npz")["val_idxs"]]
    P = ASECalculator(d).batch_eval(va)
    Fr = np.concatenate([a.get_forces() for a in va]); Fp = np.concatenate([p.get_forces() for p in P])
    nr, err = np.linalg.norm(Fr, axis=1), np.linalg.norm(Fp - Fr, axis=1)
    cells = [f"{np.mean(err[(nr >= lo) & (nr < hi)]):.2f} (n={np.sum((nr >= lo) & (nr < hi))})" for lo, hi in zip(bins[:-1], bins[1:])]
    print(f"{m:9s} | {np.median(err):.3f} | " + " | ".join(cells), flush=True)
