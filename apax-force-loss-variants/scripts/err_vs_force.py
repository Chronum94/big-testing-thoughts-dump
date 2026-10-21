"""Force error vs reference force magnitude, equal-count bins, for several LGPS students on the teacher-MD test set.

usage: python err_vs_force.py OUT.png TEST.extxyz run [run ...]   (runs under /home/chronum/scratch/lgps/models)
Per atom: |F_ref| and the error vector norm |F_pred - F_ref|. Atoms are split into N_BINS equal-count bins of |F_ref|
(the same edges for every model). Plots: absolute error vs |F| bin centre, relative error |dF|/|F_ref|, and the
ratio of each model's binned error to the first run's (the reference). Also prints a per-element table.
Evaluation uses ASECalculator.batch_eval on CPU (padded: no per-frame recompiles).
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/chronum/scratch/common")
import cpuload  # noqa: E402,F401

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from apax.md import ASECalculator  # noqa: E402
from ase.io import read  # noqa: E402

N_BINS = 10
out, test, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
frames = read(test, ":")
Fref = np.concatenate([a.get_forces() for a in frames]); sym = np.concatenate([a.get_chemical_symbols() for a in frames])
mag = np.linalg.norm(Fref, axis=1)
edges = np.quantile(mag, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
b = np.digitize(mag, edges[1:-1]); centres = np.array([np.median(mag[b == k]) for k in range(N_BINS)])
cols = ["#1f4e8c", "#d1495b", "#2e933c", "#edae49"]
res = {}
for r in runs:
    pred = ASECalculator(f"/home/chronum/scratch/lgps/models/{r}").batch_eval([a.copy() for a in frames], batch_size=32, silent=True)
    err = np.linalg.norm(np.concatenate([p.calc.results["forces"] for p in pred]) - Fref, axis=1)
    res[r] = err
    print(f"\n{r}: |dF| mean {1000 * err.mean():.1f} meV/A, median rel {np.median(err / mag):.3f}")
    print("  per element |dF| (meV/A):", {e: round(1000 * float(err[sym == e].mean()), 1) for e in sorted(set(sym))})
    print("  per bin |dF| (meV/A):", [round(1000 * float(err[b == k].mean()), 1) for k in range(N_BINS)])

fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
ref = runs[0]
for r, c in zip(runs, cols):
    m = np.array([res[r][b == k].mean() for k in range(N_BINS)]) * 1000
    rel = np.array([np.median(res[r][b == k] / mag[b == k]) for k in range(N_BINS)])
    ratio = m / (np.array([res[ref][b == k].mean() for k in range(N_BINS)]) * 1000)
    ax[0].plot(centres, m, "o-", color=c, ms=4, lw=1.4, label=r)
    ax[1].plot(centres, 100 * rel, "o-", color=c, ms=4, lw=1.4, label=r)
    ax[2].plot(centres, ratio, "o-", color=c, ms=4, lw=1.4, label=r)
for a in ax:
    a.set_xscale("log"); a.set_xlabel("|F_ref| bin median (eV/Å), equal-count bins"); a.grid(alpha=.3); a.legend(fontsize=7)
ax[0].set_title("mean |ΔF| per bin (meV/Å), lower = better", fontsize=10)
ax[1].set_title("median |ΔF| / |F_ref| per bin (%), lower = better", fontsize=10)
ax[2].set_title(f"error ratio vs {ref}, < 1 = better", fontsize=10); ax[2].axhline(1, color="0.4", lw=0.8)
fig.suptitle(f"LGPS force error vs force magnitude, teacher-MD test set ({len(mag)} atoms, {N_BINS} equal-count bins)", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=110)
