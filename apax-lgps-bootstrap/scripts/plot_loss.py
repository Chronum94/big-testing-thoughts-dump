"""Train/val E and F MAE vs epoch for the LGPS students. usage: python plot_loss.py run [run ...]"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
cols = ["#1f5fa8", "#d1642b"]
for r, c in zip(sys.argv[1:], cols):
    rows = list(csv.DictReader(open(f"models/{r}/log.csv")))
    ep = [int(x["epoch"]) for x in rows]
    for ax, key, s, unit in ((axes[0], "forces_mae", 1000, "meV/Å"), (axes[1], "energy_mae", 1000 / 50, "meV/atom")):
        ax.plot(ep, [float(x["val_" + key]) * s for x in rows], c=c, lw=1.6, label=f"{r} val (EMA)")
        ax.plot(ep, [float(x["train_" + key]) * s for x in rows], c=c, lw=0.8, ls="--", alpha=0.6, label=f"{r} train")
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel(f"{key.split('_')[0]} MAE ({unit}), lower = better")
        ax.grid(alpha=0.25, which="both")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("LGPS bootstrap (A+B+D, 738 train / 82 val), packed B5, r 6 Å")
fig.savefig("loss.png", dpi=120)
