"""Train-set MAE (same metric as eval_mad5k.py) for mad5k_<variant>_s<seed>, mean +- std over seeds."""
import numpy as np
from ase.io import read
from apax.data.statistics import PerElementRegressionShift as R
from apax.md import ASECalculator

A = read("mad_dimtri_5000.traj", ":")
variants, seeds = ["dense", "r8", "r16", "r32"], range(1, 6)
res = {v: [] for v in ["linear+0F"] + variants}
for s in seeds:
    tr = [A[i] for i in np.load(f"models/mad5k_dense_s{s}/train_val_idxs.npz")["train_idxs"]]
    Eref = np.array([a.get_potential_energy() for a in tr]); Fref = np.concatenate([a.get_forces() for a in tr])
    sh = R.compute({"numbers": [a.numbers for a in tr], "n_atoms": [len(a) for a in tr]},
                   {"energy": list(Eref)}, {"energy_regularisation": 1e-6})
    res["linear+0F"].append([np.abs(np.array([sh[a.numbers].sum() for a in tr]) - Eref).mean(), np.abs(Fref).mean()])
    for v in variants:
        P = ASECalculator(f"models/mad5k_{v}_s{s}").batch_eval(tr)
        E = np.array([p.get_potential_energy() for p in P]); F = np.concatenate([p.get_forces() for p in P])
        res[v].append([np.abs(E - Eref).mean(), np.abs(F - Fref).mean()])
    print(f"seed {s} done", flush=True)
for v, r in res.items():
    r = np.array(r); m, sd = r.mean(0), r.std(0, ddof=1)
    print(f"{v:10s} | train E {m[0]:.2f}±{sd[0]:.2f}  train F {m[1]:.2f}±{sd[1]:.2f}", flush=True)
