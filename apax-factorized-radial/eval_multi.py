"""train / val-seen / val-unseen-pair MAE, mean +- std over seeds. usage: eval_multi.py seeds model1 model2 ...
models are mad5k_<model>_s<seed>; each uses its own saved split."""
import sys
from itertools import combinations_with_replacement as cwr
import numpy as np
from ase.io import read
from apax.md import ASECalculator

import os
TAG = os.environ.get("TAG", "5k")
A = read({"2k5": "mad_dimtri.traj", "5k": "mad_dimtri_5000.traj", "5kbs8": "mad_dimtri_5000.traj", "10k": "mad_dimtri_10000.traj", "20k": "mad_dimtri_20000.traj"}[TAG], ":")
seeds, models = [int(x) for x in sys.argv[1].split(",")], sys.argv[2:]
pairs = lambda a: set(cwr(sorted(set(a.numbers)), 2))

def mae(calc, S):
    P = calc.batch_eval(S)
    Eref = np.array([a.get_potential_energy() for a in S]); Fref = np.concatenate([a.get_forces() for a in S])
    E = np.array([p.get_potential_energy() for p in P]); F = np.concatenate([p.get_forces() for p in P])
    return [np.abs(E - Eref).mean(), np.abs(F - Fref).mean()]

print(f"{'model':10s} | train E     train F     | seen E      seen F      | unseen E    unseen F")
for m in models:
    rows = []
    for s in seeds:
        d = f"models/mad{TAG}_{m}_s{s}"; idx = np.load(f"{d}/train_val_idxs.npz")
        tr, va = [A[i] for i in idx["train_idxs"]], [A[i] for i in idx["val_idxs"]]
        seen = set().union(*(pairs(a) for a in tr))
        calc = ASECalculator(d)
        rows.append(mae(calc, tr) + mae(calc, [a for a in va if pairs(a) <= seen]) + mae(calc, [a for a in va if not pairs(a) <= seen]))
    r = np.array(rows); mu, sd = r.mean(0), r.std(0, ddof=1)
    print(f"{m:10s} | " + "  ".join(f"{mu[k]:.2f}±{sd[k]:.2f}" + (" |" if k in (1, 3) else "") for k in range(6)), flush=True)
