"""Heatmaps of the crossed augmentation evaluation (arms x test sets), one panel per metric.
Colour = value / best value in that test column (1 = best; sequential, light -> dark = worse); cell text = absolute value."""
import re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E702
import numpy as np
arms, tests = ["dup", "aff", "fresh", "scan", "ortho", "gen"], ["md", "scan", "ortho", "gen"]
tl = {"md": "test MD", "scan": "isotropic\nscans", "ortho": "ortho\nscans", "gen": "general\nscans"}
def get(a, t):
    s = open(f"logs_eval/{a}_{t}.txt").read()
    m = re.search(r"^\| all \| \d+ \| ([\d.]+) \([\d.]+\) \| ([\d.]+) \([\d.]+\) \| ([\d.]+)", s, re.M)
    md = re.search(r"MAE ([\d.]+) meV/atom", s); e = [float(x) for x in re.findall(r"^\| [\d.]+ \| \d+ \| [-+\d.]+ \| ([\d.]+) \|", s, re.M)]
    return [float(x) for x in m.groups()] + [float(md.group(1)) if md else np.mean(e)]
V = np.array([[get(a, t) for t in tests] for a in arms])  # arms x tests x metrics
names = ["translational force RMSE (meV/Å)", "rotational force RMSE (meV/Å)", "intramolecular force RMSE (meV/Å)", "energy MAE (meV/atom)"]
fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
for k, nm in enumerate(names):
    M = V[:, :, k]; R = M / M.min(axis=0, keepdims=True)
    im = ax[k].imshow(R, cmap="Blues", vmin=1.0, vmax=min(R.max(), 2.2), aspect="auto")
    for i in range(len(arms)):
        for j in range(len(tests)):
            ax[k].text(j, i, f"{M[i, j]:.1f}" if k < 3 else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                       color="white" if R[i, j] > 1.0 + 0.55 * (min(R.max(), 2.2) - 1) else "black", fontweight="bold" if R[i, j] == 1 else "normal")
    ax[k].set_xticks(range(len(tests)), [tl[t] for t in tests], fontsize=9); ax[k].set_yticks(range(len(arms)), arms)
    ax[k].set_title(nm, fontsize=10); fig.colorbar(im, ax=ax[k], fraction=0.046, pad=0.03, label="÷ best in column")
fig.suptitle("EC:EMC/LiPF6 augmentation arms x held-out s2 test sets (one seed; bold = best in column; darker = worse)", y=1.02)
fig.tight_layout(); fig.savefig("crossed_heatmap.png", dpi=130, bbox_inches="tight"); print("wrote")
