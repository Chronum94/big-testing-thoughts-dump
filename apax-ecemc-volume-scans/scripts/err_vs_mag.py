"""Force-component error vs component magnitude in equal-count bins, per arm (test MD).
usage: python err_vs_mag.py TEST.extxyz OUT.png ARM [ARM ...]  (predictions from pred_cache/ or computed)
Per atom: teacher component vector t (trans / rot / intra), error vector e = student - teacher. 12 equal-count bins of |t|;
plotted: RMS |e| per bin (top row) and the ratio to the first arm (bottom row)."""
import os
import sys
sys.path.insert(0, "/home/chronum/scratch/common"); import cpuload  # noqa: E401,F401
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E402,E702
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from apax.md import ASECalculator  # noqa: E402
exec("import collections\nimport numpy as np\ndef whole" + open("/home/chronum/scratch/ecemc/volume_scan.py").read().split("def whole")[1].split("\n\n\n")[0])  # noqa: S102
exec("def decompose" + open("/home/chronum/scratch/ecemc/eval_inter_intra.py").read().split("def decompose")[1].split("calc = ASECalculator")[0])  # noqa: S102
test, out, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
frames = read(test, ":"); comps = ("trans", "rot", "intra"); tag = os.path.basename(test).replace(".extxyz", "")
T = {c: np.concatenate([decompose(a, a.get_forces(), a.arrays["mol_id"])[c] for a in frames]) for c in comps}
cols = {"dup": "#7f7f7f", "scan": "#2a6fdb", "aff": "#e0672b", "fresh": "#2ca02c", "ortho": "#9467bd", "gen": "#d62728"}
fig, ax = plt.subplots(2, 3, figsize=(15, 7.5), sharex="col"); base = {}
for arm in arms:
    cache = f"pred_cache/{arm}_{tag}.npz"; os.makedirs("pred_cache", exist_ok=True)
    if os.path.exists(cache):
        Fp = list(np.load(cache, allow_pickle=True)["F"])
    else:
        Fp = [p.get_forces() for p in ASECalculator(f"models/arm_{arm}").batch_eval(frames, batch_size=8, silent=True)]; np.savez(cache, F=np.array(Fp, dtype=object))
    S = {c: np.concatenate([decompose(a, fp, a.arrays["mol_id"])[c] for a, fp in zip(frames, Fp)]) for c in comps}
    for j, c in enumerate(comps):
        m = np.linalg.norm(T[c], axis=1); e = np.linalg.norm(S[c] - T[c], axis=1); keep = m > 1e-9
        edges = np.quantile(m[keep], np.linspace(0, 1, 13)); idx = np.clip(np.digitize(m[keep], edges[1:-1]), 0, 11)
        x = np.array([np.median(m[keep][idx == k]) for k in range(12)]); y = np.array([np.sqrt(np.mean(e[keep][idx == k] ** 2)) for k in range(12)])
        ax[0, j].plot(1e3 * x, 1e3 * y, "o-", color=cols.get(arm), lw=1.5, ms=4, label=arm)
        if arm == arms[0]:
            base[c] = y
        ax[1, j].plot(1e3 * x, y / base[c], "o-", color=cols.get(arm), lw=1.5, ms=4, label=arm)
    print(arm, "done", flush=True)
for j, c in enumerate(comps):
    ax[0, j].set_title(f"{c}: RMS error vs |teacher component| (12 equal-count bins)", fontsize=10); ax[0, j].set_ylabel("RMS |error| (meV/Å)")
    ax[1, j].set_ylabel(f"ratio to {arms[0]}"); ax[1, j].axhline(1, color="0.5", lw=0.8); ax[1, j].set_xlabel("|teacher component| (meV/Å, bin median)")
    for a in ax[:, j]:
        a.set_xscale("log"); a.spines[["top", "right"]].set_visible(False); a.legend(frameon=False, fontsize=8)
fig.suptitle(f"force-component error vs magnitude, {test} ({len(frames)} frames)"); fig.tight_layout(); fig.savefig(out, dpi=120); print("wrote", out)
