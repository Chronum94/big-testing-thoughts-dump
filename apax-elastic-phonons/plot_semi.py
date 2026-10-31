"""Student vs teacher elastic constants (27 cubic semiconductors) -> elastic_semi.png
Top: parity (log-log) for C11, C12, C44, B, relaxed-ion. Bottom: C44 split into clamped-ion and internal-relaxation parts.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

OI = {"v3": "#E69F00", "v4": "#0072B2", "grey": "#999999"}  # Okabe-Ito
L = lambda f: {r["name"]: r for r in map(json.loads, open(f))}  # noqa: E731
T = L("semi_teacher.jsonl")
S = {"v3": L("semi_seed_v3.jsonl"), "v4": L("semi_seed_v4.jsonl")}
names = list(T)

fig = plt.figure(figsize=(16, 9.5), layout="constrained")
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
for j, k in enumerate(("C11", "C12", "C44", "B")):
    ax = fig.add_subplot(gs[0, j])
    t = np.array([T[n]["relaxed"][k] for n in names])
    lo, hi = max(t.min() * 0.5, 0.5), t.max() * 1.6
    ax.fill_between([lo, hi], [lo / 1.1, hi / 1.1], [lo * 1.1, hi * 1.1], color=OI["grey"], alpha=0.15, lw=0, label="±10 %")
    ax.plot([lo, hi], [lo, hi], color=OI["grey"], lw=1)
    for m, mk in (("v3", "o"), ("v4", "s")):
        s = np.array([S[m][n]["relaxed"][k] for n in names])
        ax.scatter(t, s, s=34, marker=mk, facecolor="none" if m == "v4" else OI[m], edgecolor=OI[m], lw=1.4,
                   label=f"seed_{m}", zorder=3)
    for n in (("SnTe", "C") if k in ("C11", "B") else ("SnTe", "PbTe", "C")):  # selective labels: outliers + the stiffest
        ax.annotate(n, (T[n]["relaxed"][k], S["v3"][n]["relaxed"][k]), xytext=(4, -10), textcoords="offset points", fontsize=8)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel=f"teacher {k} (GPa)", ylabel=f"student {k} (GPa)",
           title=f"{k}, relaxed-ion")
    ax.set_aspect("equal")
    if j == 0:
        ax.legend(loc="upper left", fontsize=9, frameon=False)

ax = fig.add_subplot(gs[1, :])
sel = [n for n in names if T[n]["clamped"]["C44"] - T[n]["relaxed"]["C44"] > 0.05 * T[n]["clamped"]["C44"]]
x = np.arange(len(sel))
for i, (m, part) in enumerate([(m, p) for m in ("v3", "v4") for p in ("clamped", "drop")]):
    v = []
    for n in sel:
        tc, tr = T[n]["clamped"]["C44"], T[n]["relaxed"]["C44"]
        sc, sr = S[m][n]["clamped"]["C44"], S[m][n]["relaxed"]["C44"]
        v.append((sc / tc - 1) * 100 if part == "clamped" else ((sc - sr) / (tc - tr) - 1) * 100)
    ax.bar(x + (i - 1.5) * 0.2, v, 0.18, color=OI[m], alpha=0.45 if part == "clamped" else 1.0,
           hatch="//" if part == "clamped" else None, edgecolor=OI[m], lw=0.8,
           label=f"seed_{m}: {'clamped-ion C44 (stress only)' if part == 'clamped' else 'internal-relaxation drop (forces)'}")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, sel, rotation=45, ha="right")
ax.set_ylabel("student vs teacher (%)")
ax.set_ylim(-30, 60)
if "SnTe" in sel:  # rocksalt: teacher off-centres under shear, students not at all
    ax.annotate("SnTe drop −100 %:\nteacher ions off-centre,\nstudents don't", (sel.index("SnTe") + 0.2, -29), xytext=(-150, 25), textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_title("C44 = clamped-ion part − internal-relaxation drop: the stress part is right, the sublattice relaxation is 16–23 % too large "
             "(compounds with drop > 5 % of C44)")
ax.legend(ncol=2, fontsize=9, frameon=False, loc="upper left")
ax.grid(axis="y", color=OI["grey"], alpha=0.3)
fig.suptitle("apax students vs PET-MAD-m teacher: elastic constants of 27 cubic semiconductors", fontsize=14)
fig.savefig("elastic_semi.png", dpi=150)
print("wrote elastic_semi.png")
