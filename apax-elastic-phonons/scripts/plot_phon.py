"""Phonon bands, teacher vs students, 27 semiconductors -> phonons_semi.png; mode table -> phon_table.md"""
import json

import matplotlib.pyplot as plt
import numpy as np

OI = {"teacher": "#000000", "seed_v3": "#E69F00", "seed_v4": "#0072B2"}
STY = {"teacher": "-", "seed_v3": "--", "seed_v4": ":"}
names = [json.loads(line)["name"] for line in open("/home/chronum/scratch/elastic/semi_teacher.jsonl")]
B = {m: {n: np.load(f"bands/{m}/{n}.npz") for n in names} for m in OI}

fig, axs = plt.subplots(4, 7, figsize=(22, 13), layout="constrained")
for ax, n in zip(axs.flat, names):
    for m in OI:
        d = B[m][n]
        x = d["x"] / d["x"][-1]  # each model's own path, normalised (lattice constants differ by < 0.6 %)
        ax.plot(x, d["E"], STY[m], color=OI[m], lw=1.3 if m == "teacher" else 1.1)
    d = B["teacher"][n]
    X = d["X"] / d["x"][-1]
    for xx in X:
        ax.axvline(xx, color="#999999", lw=0.5)
    ax.set_xticks(X, [s.replace("G", "Γ") for s in d["labels"]])
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=min(0, min(B[m][n]["E"].min() for m in OI)) - 0.2)
    ax.set_title(n, fontsize=11)
    ax.tick_params(labelsize=8)
for ax in axs[:, 0]:
    ax.set_ylabel("THz")
axs.flat[-1].axis("off")
for m in OI:
    axs.flat[-1].plot([], [], STY[m], color=OI[m], lw=2, label=m if m != "teacher" else "PET-MAD-m teacher")
axs.flat[-1].legend(loc="center", fontsize=13, frameon=False)
fig.suptitle("Phonon bands (ndsc, 6×6×6 q-mesh, each model at its own relaxed a; no LO–TO splitting): apax students vs PET-MAD-m", fontsize=15)
fig.savefig("phonons_semi.png", dpi=130)

# mode table: Gamma optical (top band at G), X TA (lowest), X LA/LO top, full-path band RMS (on teacher's normalised x grid)
rows, rel = [], {m: {"G_opt": [], "X_TA": [], "rms": []} for m in ("seed_v3", "seed_v4")}
for n in names:
    t = B["teacher"][n]
    r = [n, f"{t['G'][-1]:.2f}", f"{t['Xp'][0]:.2f}"]
    for m in ("seed_v3", "seed_v4"):
        s = B[m][n]
        g, xta = (s["G"][-1] / t["G"][-1] - 1) * 100, (s["Xp"][0] / t["Xp"][0] - 1) * 100
        rms = np.sqrt(np.mean((s["E"] - t["E"]) ** 2))  # same npoints/path in fractional q, so index-aligned
        rel[m]["G_opt"].append(g); rel[m]["X_TA"].append(xta); rel[m]["rms"].append(rms)
        r += [f"{g:+.1f}", f"{xta:+.1f}", f"{rms:.2f}"]
    rows.append(r)
with open("phon_table.md", "w") as f:
    f.write("| material | teacher Γ-opt (THz) | teacher X-TA (THz) | v3 Γ-opt Δ% | v3 X-TA Δ% | v3 band RMS (THz) | v4 Γ-opt Δ% | v4 X-TA Δ% | v4 band RMS (THz) |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| " + " | ".join(r) + " |\n")
    f.write("| **median** | | | " + " | ".join(f"{np.median(rel[m][k]):+.1f}" if k != "rms" else f"{np.median(rel[m][k]):.2f}"
                                          for m in rel for k in ("G_opt", "X_TA", "rms")) + " |\n")
print(open("phon_table.md").read().splitlines()[-1])
json.dump({m: {k: list(map(float, v)) for k, v in d.items()} for m, d in rel.items()}, open("phon_rel.json", "w"))
