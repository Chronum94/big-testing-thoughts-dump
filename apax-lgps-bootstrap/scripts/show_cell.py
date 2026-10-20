"""Render the lowest-energy ordered LGPS cell: projections along a and c, PS4/GeS4 bonds drawn, Li by site."""
import sys

import matplotlib
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

a = read(sys.argv[1], 0).repeat((2, 2, 1))
sym = np.array(a.get_chemical_symbols())
col = {"Li": "#2a9d8f", "Ge": "#7b2cbf", "P": "#e76f51", "S": "#e9c46a"}
size = {"Li": 40, "Ge": 120, "P": 90, "S": 70}
i, j, D = neighbor_list("ijD", a, 2.6)
bond = (np.isin(sym[i], ["P", "Ge"])) & (sym[j] == "S")
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, (x, y, lab) in zip(axes, [(0, 2, "view along b (x–z)"), (0, 1, "view along c (x–y)")]):
    R = a.positions
    for k in np.flatnonzero(bond):
        p, q = R[i[k]], R[i[k]] + D[k]
        ax.plot([p[x], q[x]], [p[y], q[y]], c="#555", lw=1.2, zorder=1)
    for s in ["S", "P", "Ge", "Li"]:
        m = sym == s
        ax.scatter(R[m, x], R[m, y], s=size[s], c=col[s], edgecolors="k", linewidths=0.4, zorder=2, label=s)
    c = a.cell
    corners = np.array([[0, 0], [c[0, x] + c[1, x] * (y == 1), 0], [0, 0]])
    ax.set_aspect("equal"); ax.set_title(lab); ax.set_xlabel("Å"); ax.set_ylabel("Å")
axes[0].legend(loc="upper right", fontsize=9)
c = read(sys.argv[1], 0).cell.cellpar()
fig.suptitle(f"Li10GeP2S12, lowest PET-MAD-m ordering, 2×2×1 of the 50-atom cell (a {c[0]:.3f}, b {c[1]:.3f}, c {c[2]:.3f} Å)")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110)
