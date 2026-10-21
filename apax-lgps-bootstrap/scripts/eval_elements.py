"""Per-element force errors of LGPS students on val (by bootstrap block) and teacher-MD test frames; bar plot.

usage: python eval_elements.py run [run ...]
"""
import sys

sys.path.insert(0, "/home/chronum/scratch/lgps")
import cpuload  # noqa: E402,F401
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from apax.md import ASECalculator  # noqa: E402
from ase.io import read  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EL = ["Li", "Ge", "P", "S"]
groups = {"val: elastic": ("val", ["A_elastic"]), "val: Li disorder": ("val", ["B_li_disorder", "B_li_start"]),
          "val: rattle + volume": ("val", ["D_rattle", "D_volume"]),
          "test: MD 600 K": ("test", ["test_md_600K"]), "test: MD 900 K": ("test", ["test_md_900K"])}
sets = {"val": read("data/val.extxyz", ":"), "test": read("data/test.extxyz", ":")}
res = {}
for r in sys.argv[1:]:
    calc = ASECalculator(f"models/{r}")
    for g, (s, cats) in groups.items():
        err = {e: [] for e in EL}; ref = {e: [] for e in EL}
        for a in sets[s]:
            if a.info["category"] not in cats:
                continue
            F = a.get_forces(); b = a.copy(); b.calc = calc; dF = np.abs(b.get_forces() - F)
            sym = np.array(a.get_chemical_symbols())
            for e in EL:
                err[e].append(dF[sym == e].ravel()); ref[e].append(np.abs(F[sym == e]).ravel())
        res[(r, g)] = {e: (1000 * np.concatenate(err[e]).mean(), 1000 * np.concatenate(ref[e]).mean()) for e in EL}
        print(r, g, " ".join(f"{e} {v[0]:.1f} ({100 * v[0] / v[1]:.0f} % of |F| {v[1]:.0f})" for e, v in res[(r, g)].items()), flush=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
cols = ["#1f5fa8", "#5b8fd1", "#9cb9e3", "#d1642b", "#8a3b12"]
r = sys.argv[1]; w = 0.16; x = np.arange(len(EL))
for k, (g, c) in enumerate(zip(groups, cols)):
    v = res[(r, g)]
    axes[0].bar(x + (k - 2) * w, [v[e][0] for e in EL], w, color=c, label=g)
    axes[1].bar(x + (k - 2) * w, [100 * v[e][0] / v[e][1] for e in EL], w, color=c, label=g)
for ax, yl in zip(axes, ["force MAE (meV/Å), lower = better", "force MAE / mean |F| (%), lower = better"]):
    ax.set_xticks(x, EL); ax.set_ylabel(yl); ax.grid(axis="y", alpha=0.25)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle(f"LGPS {r}: force error by element (per Cartesian component)")
fig.savefig("err_by_element.png", dpi=130)
