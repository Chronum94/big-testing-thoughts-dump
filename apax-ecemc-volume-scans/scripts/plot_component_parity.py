"""Error plots (student - teacher vs teacher) of per-atom force components split into translational / rotational / intramolecular parts,
one row per arm. usage: python plot_component_parity.py TEST.extxyz OUT.png ARM [ARM ...]   (CPU)
Cartesian components pooled over atoms; red = Li+ ions (drawn underneath), grey = all atoms (random 40k subsample) on top.
RMSE in each title. Student predictions are cached in pred_cache/<arm>_<test>.npz."""
import sys

sys.path.insert(0, "/home/chronum/scratch/common"); import cpuload  # noqa: E401,F401
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E402,E702
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from apax.md import ASECalculator  # noqa: E402

src = open("/home/chronum/scratch/ecemc/eval_inter_intra.py").read()
exec("import collections\nimport numpy as np\n" + "def whole" + open("/home/chronum/scratch/ecemc/volume_scan.py").read().split("def whole")[1].split("from scipy")[0])  # noqa: S102
exec("def decompose" + src.split("def decompose")[1].split("calc = ASECalculator")[0])  # noqa: S102
test, out, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
frames = read(test, ":"); comps = ("trans", "rot", "intra")
T = {c: [] for c in comps}; isli = []
for a in frames:
    d = decompose(a, a.get_forces(), a.arrays["mol_id"])
    for c in comps:
        T[c].append(d[c])
    mol = a.arrays["mol_id"]; isli.append(np.array([(mol == m).sum() == 1 and s == "Li" for m, s in zip(mol, a.get_chemical_symbols())]))
T = {c: np.concatenate(v).ravel() for c, v in T.items()}; isli = np.repeat(np.concatenate(isli), 3)
sub = np.random.default_rng(0).choice(len(isli), min(40000, len(isli)), replace=False)
YLIM = {"trans": 0.5, "rot": 0.12, "intra": 0.6}  # shared across arms so rows are comparable
fig, ax = plt.subplots(len(arms), 3, figsize=(12, 3.8 * len(arms)), squeeze=False)
for r, arm in enumerate(arms):
    import os
    cache = f"pred_cache/{arm}_{os.path.basename(test).replace('.extxyz', '')}.npz"; os.makedirs("pred_cache", exist_ok=True)
    if os.path.exists(cache):
        Fp = list(np.load(cache, allow_pickle=True)["F"])
    else:
        Fp = [p.get_forces() for p in ASECalculator(f"models/arm_{arm}").batch_eval(frames, batch_size=8, silent=True)]
        np.savez(cache, F=np.array(Fp, dtype=object))
    S = {c: [] for c in comps}
    for a, fp in zip(frames, Fp):
        d = decompose(a, fp, a.arrays["mol_id"])
        for c in comps:
            S[c].append(d[c])
    for j, c in enumerate(comps):
        s = np.concatenate(S[c]).ravel(); t = T[c]; rmse = 1e3 * np.sqrt(np.mean((s - t) ** 2))
        e = s - t; lim = np.percentile(np.abs(t), 99.9) * 1.1; ylim = YLIM[c]
        ax[r, j].scatter(t[isli], e[isli], s=4, color="#d62728", alpha=0.6, zorder=1, rasterized=True,
                         label=f"Li+ ({1e3 * np.sqrt(np.mean(e[isli] ** 2)):.0f})" if c == "trans" else None)
        ax[r, j].scatter(t[sub], e[sub], s=1, color="0.4", alpha=0.25, zorder=2, rasterized=True)
        ax[r, j].axhline(0, color="0.2", lw=0.6, zorder=3); ax[r, j].set_xlim(-lim, lim); ax[r, j].set_ylim(-ylim, ylim)
        xs = np.array([-lim, lim])  # least-squares line err = a + b * teacher on all points (not the subsample); b < 0 = shrinkage
        for sel, col, nm in ((np.ones_like(isli, dtype=bool), "k", "all"), (isli, "#d62728", "Li+")):
            if sel.sum() > 10 and (nm == "all" or c == "trans"):
                b, a0 = np.polyfit(t[sel], e[sel], 1)
                ax[r, j].plot(xs, a0 + b * xs, color=col, lw=1.6, ls="--" if nm == "Li+" else "-", zorder=4, label=f"fit {nm}: slope {b:+.3f}")
        ax[r, j].legend(frameon=False, fontsize=7, loc="upper left")
        ax[r, j].set_title(f"{arm}: {c}  RMSE {rmse:.1f} meV/Å", fontsize=10)
        ax[r, j].set_xlabel("teacher (eV/Å)"); ax[r, j].set_ylabel("student − teacher (eV/Å)")
    print(arm, "done", flush=True)
fig.suptitle(f"force-component errors, {test} ({len(frames)} frames); grey: all atoms (40k subsample), red: Li+ (underneath)", y=1.0)
fig.tight_layout(); fig.savefig(out, dpi=110); print("wrote", out)
