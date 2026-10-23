"""Learning curves of the four intervention arms (shared s2 val set): val E MAE, val F MAE, train and val loss."""
import csv
import re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E702
import numpy as np
import sys
ALL = {"dup": "#7f7f7f", "scan": "#2a6fdb", "aff": "#e0672b", "fresh": "#2ca02c", "ortho": "#9467bd", "gen": "#d62728", "genroll": "#17becf", "genroll_pbp": "#bcbd22",
       "g03": "#fc9272", "g12": "#a50f15", "g20": "#3f0008", "o03": "#c5b0d5", "o12": "#6a3d9a", "o20": "#2d0f4f"}  # strain ladders: darker = larger norm
arms = sys.argv[2:] or ["dup", "scan", "aff", "fresh"]
VAR = {"": "#d62728", "_r5t6": "#ff7f0e", "_r6t5": "#2a6fdb", "_r6": "#9467bd"}  # cutoff / tensor-channel variants
def colour(a):
    b = re.sub(r"_s\d$", "", a); v = next((k for k in ("_r5t6", "_r6t5", "_r6") if b.endswith(k)), "")
    return ALL.get(b) or VAR[v]
cols = {a: colour(a) for a in arms}; OUT = sys.argv[1] if len(sys.argv) > 1 else "arms_loss.png"
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
for arm, c in cols.items():
    r = list({x["epoch"]: x for x in csv.DictReader(open(f"models/arm_{arm}/log.csv"))}.values())  # restarted runs repeat epochs: keep the last
    g = lambda k: np.array([float(x[k]) for x in r])  # noqa: E731
    e = g("epoch")
    ax[0].plot(e, 1e3 * g("val_energy_mae") / 143, color=c, lw=1.5, ls="--" if arm.endswith("_s2") else "-", label=f"{arm}: {1e3 * g('val_energy_mae')[-1] / 143:.2f}")
    ax[1].plot(e, 1e3 * g("val_forces_mae"), color=c, lw=1.5, ls="--" if arm.endswith("_s2") else "-", label=f"{arm}: {1e3 * g('val_forces_mae')[-1]:.1f}")
    ax[2].plot(e, g("train_loss"), color=c, lw=1.5, ls="--" if arm.endswith("_s2") else "-", label=f"{arm}: {g('train_loss')[-1]:.3g}")
    ax[3].plot(e, g("val_loss"), color=c, lw=1.5, ls="--" if arm.endswith("_s2") else "-", label=f"{arm}: {g('val_loss')[-1]:.3g}")
for a, t in zip(ax, ("val E MAE (meV/atom, ~143 atoms)", "val F MAE (meV/Å)", "train loss", "val loss")):
    a.set_yscale("log"); a.set_xlabel("epoch"); a.set_title(t); a.legend(frameon=False, fontsize=8, title="final")
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("EC:EMC/LiPF6 arms (same size, same recipe; val = held-out s2 MD frames; EMA from epoch 100)")
fig.tight_layout(); fig.savefig(OUT, dpi=130); print("wrote")
