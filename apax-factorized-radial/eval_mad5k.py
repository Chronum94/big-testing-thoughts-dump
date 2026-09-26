"""Val MAE on seen vs unseen-pair frames for mad5k_<variant>_s<seed>, mean +- std over seeds."""
import sys
from itertools import combinations_with_replacement as cwr
import numpy as np
from ase.io import read
from apax.data.statistics import PerElementRegressionShift as R
from apax.md import ASECalculator

A = read("mad_dimtri_5000.traj", ":")
variants, seeds = ["dense", "r8", "r16", "r32"], range(1, 6)
pairs = lambda a: set(cwr(sorted(set(a.numbers)), 2))

def mae(E, F, Eref, Fref, w):
    return (np.abs(E[w] - Eref[w]).mean(),
            np.mean(np.concatenate([np.abs(F[i] - Fref[i]).ravel() for i in w])))

res = {v: [] for v in ["linear+0F"] + variants}
for s in seeds:
    idx = np.load(f"models/mad5k_dense_s{s}/train_val_idxs.npz")
    tr, va = [A[i] for i in idx["train_idxs"]], [A[i] for i in idx["val_idxs"]]
    seen = set().union(*(pairs(a) for a in tr))
    unseen = np.array([not pairs(a) <= seen for a in va])
    groups = (np.where(~unseen)[0], np.where(unseen)[0])
    Eref = np.array([a.get_potential_energy() for a in va]); Fref = [a.get_forces() for a in va]
    sh = R.compute({"numbers": [a.numbers for a in tr], "n_atoms": [len(a) for a in tr]},
                   {"energy": [a.get_potential_energy() for a in tr]}, {"energy_regularisation": 1e-6})
    E0 = np.array([sh[a.numbers].sum() for a in va])
    res["linear+0F"].append([*mae(E0, [np.zeros_like(f) for f in Fref], Eref, Fref, groups[0]),
                             *mae(E0, [np.zeros_like(f) for f in Fref], Eref, Fref, groups[1])])
    for v in variants:
        P = ASECalculator(f"models/mad5k_{v}_s{s}").batch_eval(va)
        E = np.array([p.get_potential_energy() for p in P]); F = [p.get_forces() for p in P]
        res[v].append([*mae(E, F, Eref, Fref, groups[0]), *mae(E, F, Eref, Fref, groups[1])])
    print(f"seed {s}: {len(groups[0])} seen / {len(groups[1])} unseen-pair val frames", flush=True)

print(f"{'model':10s} | seen E        seen F        | unseen E      unseen F")
for v, r in res.items():
    r = np.array(r); m, sd = r.mean(0), r.std(0, ddof=1)
    print(f"{v:10s} | " + "  ".join(f"{m[k]:.2f}±{sd[k]:.2f}" + (" |" if k == 1 else "") for k in range(4)), flush=True)
