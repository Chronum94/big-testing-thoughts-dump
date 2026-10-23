"""Energy and force error scatters for matched runs without / with the pairwise energy-difference loss.

usage: python plot_pw_scatter.py OUT.png ARM_A:ARM_B [ARM_A:ARM_B ...]   (CPU; ARM = model name without arm_)
One row per pair; columns: E error of A, E error of B, F error of A, F error of B.
E: per-frame (pred - teacher) / atom vs teacher E / atom, coloured by test set, with per-set MAE and mean offset.
F: Cartesian force-component error vs teacher component (all atoms, 30k subsample), with RMSE and linear-fit slope.
Test sets: md, shift, x2. Predictions are cached in pred_cache/<arm>_<set>_EF.npz.
"""
import os
import sys

sys.path.insert(0, "/home/chronum/scratch/common"); import cpuload  # noqa: E401,F401
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E402,E702
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

SETS = {"md": "#7f7f7f", "shift": "#2a6fdb", "x2": "#e0672b"}
out, pairs = sys.argv[1], [p.split(":") for p in sys.argv[2:]]
frames = {t: read(f"data/test_{t}.extxyz", ":") for t in SETS}
ref = {t: (np.array([a.get_potential_energy() / len(a) for a in fr]), np.concatenate([a.get_forces().ravel() for a in fr]))
       for t, fr in frames.items()}


def predict(arm, t):
    cache = f"pred_cache/{arm}_{t}_EF.npz"; os.makedirs("pred_cache", exist_ok=True)
    if not os.path.exists(cache):
        from apax.md import ASECalculator
        pr = ASECalculator(f"models/arm_{arm}").batch_eval(frames[t], batch_size=8, silent=True)
        np.savez(cache, E=np.array([p.get_potential_energy() / len(p) for p in pr]),
                 F=np.concatenate([p.get_forces().ravel() for p in pr]))
    d = np.load(cache); return d["E"], d["F"]


rng = np.random.default_rng(0)
fig, ax = plt.subplots(len(pairs), 4, figsize=(18, 3.9 * len(pairs)), squeeze=False)
for r, pair in enumerate(pairs):
    for k, arm in enumerate(pair):
        lab = f"{arm} ({'+' if k else 'no'} pairwise)"
        ea, fa = ax[r, k], ax[r, 2 + k]
        for t, c in SETS.items():
            E, F = predict(arm, t); Et, Ft = ref[t]; e = 1e3 * (E - Et); f = F - Ft
            ea.scatter(Et, e, s=7 if t != "x2" else 18, color=c, alpha=0.6, rasterized=True,
                       label=f"{t}: MAE {np.abs(e).mean():.2f}, mean {e.mean():+.2f}")
            s = rng.choice(len(f), min(10000, len(f)), replace=False)
            slope = np.polyfit(Ft, f, 1)[0]
            fa.scatter(Ft[s], f[s], s=2, color=c, alpha=0.3, rasterized=True,
                       label=f"{t}: RMSE {1e3 * np.sqrt(np.mean(f ** 2)):.0f}, slope {slope:+.3f}")
        ea.axhline(0, color="k", lw=0.6); fa.axhline(0, color="k", lw=0.6)
        ea.set(title=f"{lab}: energy", xlabel="teacher E (eV/atom)", ylabel="student − teacher (meV/atom)", ylim=(-8, 8))
        fa.set(title=f"{lab}: forces", xlabel="teacher F component (eV/Å)", ylabel="student − teacher (eV/Å)", xlim=(-6, 6), ylim=(-0.8, 0.8))
        ea.legend(fontsize=7, loc="upper left", markerscale=2); fa.legend(fontsize=7, loc="upper left", markerscale=4)
fig.suptitle("Energy / force errors without vs with pairwise loss (λ = 1e3), seed 1; grey = test MD, blue = density/T shift, orange = 2× boxes")
fig.tight_layout(); fig.savefig(out, dpi=110); print("wrote", out)
