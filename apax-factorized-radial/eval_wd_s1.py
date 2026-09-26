"""Seed-1 split: train / val-seen / val-unseen-pair MAE for dense wd scan vs factorized."""
from itertools import combinations_with_replacement as cwr
import numpy as np
from ase.io import read
from apax.data.statistics import PerElementRegressionShift as R
from apax.md import ASECalculator

A = read("mad_dimtri_5000.traj", ":")
idx = np.load("models/mad5k_dense_s1/train_val_idxs.npz")
tr, va = [A[i] for i in idx["train_idxs"]], [A[i] for i in idx["val_idxs"]]
pairs = lambda a: set(cwr(sorted(set(a.numbers)), 2))
seen = set().union(*(pairs(a) for a in tr))
unseen = np.array([not pairs(a) <= seen for a in va])
sets = {"train": tr, "seen": [a for a, u in zip(va, unseen) if not u], "unseen": [a for a, u in zip(va, unseen) if u]}
sh = R.compute({"numbers": [a.numbers for a in tr], "n_atoms": [len(a) for a in tr]},
               {"energy": [a.get_potential_energy() for a in tr]}, {"energy_regularisation": 1e-6})

def mae(Epred, Fpred, S):
    Eref = np.array([a.get_potential_energy() for a in S]); Fref = np.concatenate([a.get_forces() for a in S])
    return np.abs(Epred - Eref).mean(), np.abs(Fpred - Fref).mean()

import sys
models = sys.argv[1:] or (["dense"] + [f"dense_wd{w}" for w in ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]] + ["r8", "r16", "r32"])
print(f"{'model':13s} | train E  train F | seen E  seen F | unseen E  unseen F")
row = []
for k, S in sets.items():
    row += mae(np.array([sh[a.numbers].sum() for a in S]), np.zeros((sum(len(a) for a in S), 3)), S)
print(f"{'linear+0F':13s} | " + "  ".join(f"{x:.2f}" for x in row), flush=True)
for m in models:
    calc = ASECalculator(f"models/mad5k_{m}_s1"); row = []
    for k, S in sets.items():
        P = calc.batch_eval(S)
        row += mae(np.array([p.get_potential_energy() for p in P]), np.concatenate([p.get_forces() for p in P]), S)
    print(f"{m:13s} | " + "  ".join(f"{x:.2f}" for x in row), flush=True)
