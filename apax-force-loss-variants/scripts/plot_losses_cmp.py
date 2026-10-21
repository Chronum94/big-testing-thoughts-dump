"""Compare LGPS runs that differ only in the force loss: val force / energy MAE vs epoch. usage: python plot_losses_cmp.py OUT.png run [run ...]"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

out, runs = sys.argv[1], sys.argv[2:]
cols = ["#1f4e8c", "#d1495b", "#2e933c", "#edae49"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for r, c in zip(runs, cols):
    p = f"/home/chronum/scratch/lgps/models/{r}/log.csv"
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    for a, k, s in ((ax[0], "val_forces_mae", 1000), (ax[1], "val_energy_mae", 1000 / 50), (ax[2], "val_forces_mse", 1e6)):
        a.plot(d.epoch, d[k] * s, color=c, lw=1.1, label=f"{r} (ep {int(d.epoch.iloc[-1])}): {d[k].iloc[-1] * s:.3g}")
for a, t in zip(ax, ("val force MAE (meV/Å)", "val energy MAE (meV/atom)", "val force MSE (meV²/Å²)")):
    a.set_yscale("log"); a.set_xlabel("epoch"); a.set_title(t + ", lower = better", fontsize=10); a.grid(alpha=.3); a.legend(fontsize=7)
fig.suptitle("LGPS, seed 1, scanned epoch: force-loss variants (MSE baseline vs Huber vs relative)", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=110)
