"""Val MAE vs MD wall time (cost map, 12,960-atom box) for the molcrys Phase-1 / L-ladder runs."""
import csv, os
import json
import sys
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# run: (label, MD ms from costmap.jsonl, family)
RUNS = {"v2B4_s1": ("L0 A12/B4", 15.65, "full l≤3"), "p1_B2": ("B2", 9.8, "full l≤3"),
        "p1_B3": ("B3", 13.3, "full l≤3"), "p1_B5": ("B5", 16.3, "full l≤3"), "p1_B6": ("B6", 24.55, "full l≤3"), "p1_B8": ("B8", 35.7, "full l≤3"),
        "v2nc_s1": ("L1 (−c4,c6)", 13.1, "contraction drops"), "v2L2_s1": ("L2 (−c4,c6,c7)", 12.3, "contraction drops"),
        "v2L3_s1": ("L3 A12/B4", 10.0, "l≤2"), "p1_L3A24": ("L3 A24", 11.4, "l≤2"),
        "p1_L3A32": ("L3 A32", 11.3, "l≤2"), "p1_L3B6": ("L3 B6", 10.8, "l≤2"), "p1_L3B10": ("L3 B10", 20.1, "l≤2"), "p1_L3B8": ("L3 B8", 15.7, "l≤2"), "p1_A16B5": ("A16 B5", 16.3, "full l≤3")}
STYLE = {"full l≤3": ("#1f5fa8", "o"), "contraction drops": ("#8c8c8c", "s"), "l≤2": ("#d1642b", "D")}

PACKED = "--packed" in sys.argv  # use packed-moments MD timings (identical predictions, see packed_equiv/)
if PACKED:
    pk = {}
    for l in open(f"{ROOT}/costmap/costmap.jsonl"):
        r = json.loads(l)
        if r.get("packed"):
            pk[(r["A"], r["B"], bool(r["drop"]))] = r["md_ms_box12960"]
    CFG = {"v2B4_s1": (12, 4, 0), "p1_B2": (12, 2, 0), "p1_B3": (12, 3, 0), "p1_B5": (12, 5, 0), "p1_B6": (12, 6, 0),
           "p1_B8": (12, 8, 0), "v2L3_s1": (12, 4, 1), "p1_L3B6": (12, 6, 1), "p1_L3B10": (12, 10, 1)}
    for r, k in CFG.items():
        lab, md, fam = RUNS[r]
        RUNS[r] = (lab, min(md, pk[k]), fam)  # L3: packing does not help, keep the faster layout
    for r in ("v2nc_s1", "v2L2_s1", "p1_L3A24", "p1_L3A32", "p1_A16B5"):  # not re-timed packed
        RUNS.pop(r, None)
pts = []
for r, (lab, md, fam) in RUNS.items():
    try:
        rows = list(csv.DictReader(open(f"{ROOT}/logs/{r}.csv")))
    except FileNotFoundError:
        continue
    if len(rows) < 400:  # ponytail: skip unfinished runs; rerun the script when they land
        continue
    pts.append((lab, md, fam, 1000 * float(rows[-1]["val_forces_mae"]), float(rows[-1]["val_energy_mae"])))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, k, name in ((axes[0], 3, "val force MAE (meV/Å)"), (axes[1], 4, "val energy MAE (eV/structure)")):
    for fam, (c, m) in STYLE.items():
        p = [q for q in pts if q[2] == fam]
        ax.scatter([q[1] for q in p], [q[k] for q in p], c=c, marker=m, s=60, label=fam, zorder=3,
                   edgecolors="white", linewidths=1.5)
    hull = []  # lower convex hull (monotone chain), cut at the most accurate point: the convex Pareto front
    for q in sorted(pts, key=lambda q: (q[1], q[k])):
        while len(hull) >= 2 and ((hull[-1][1] - hull[-2][1]) * (q[k] - hull[-2][k])
                                  - (hull[-1][k] - hull[-2][k]) * (q[1] - hull[-2][1])) <= 0:
            hull.pop()
        hull.append(q)
    hull = hull[:min(range(len(hull)), key=lambda i: hull[i][k]) + 1]
    ax.plot([q[1] for q in hull], [q[k] for q in hull], c="k", lw=1, alpha=0.5, zorder=1, label="convex Pareto front")
    for q in pts:
        ax.annotate(q[0], (q[1], q[k]), xytext={"L3 A32": (6, 6), "L3 A24": (-48, -12), "B5": (2, -14), "A16 B5": (6, 6)}.get(q[0], (6, 4)), textcoords="offset points", fontsize=8, color="#333")
    ref = next(q for q in pts if q[0].startswith("L0"))
    ax.axhline(ref[k], c="#1f5fa8", lw=0.8, ls=":", alpha=0.6)
    ax.axvline(ref[1], c="#1f5fa8", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("MD time, 12,960 atoms (ms/step)  →  slower")
    ax.set_ylabel(f"{name}  (lower = better)")
    ax.grid(alpha=0.2)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("molcrys GMNN, 400 epochs, seed 1: accuracy vs MD cost" + (" (packed moments)" if PACKED else "") + " (dotted = L0 baseline)")
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pareto_packed.png" if PACKED else "pareto.png"), dpi=150)
