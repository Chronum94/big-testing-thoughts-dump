"""Student-vs-teacher errors on AL-sampled frames (per run/temperature and per element), and selection for the next
training set: the top FRAC_TOP frames by force MAE plus FRAC_RAND random others.

usage: python al_select.py labelled.extxyz OUT_selected.extxyz [--top 0.3] [--rand 0.1] [--seed 0]
"""
import argparse
from collections import defaultdict

import numpy as np
from ase.io import read, write

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--top", type=float, default=0.3); ap.add_argument("--rand", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
a_ = ap.parse_args()
fr = read(a_.inp, ":")
rows, by_el = defaultdict(list), defaultdict(list)
fmae = []
for a in fr:
    dF = np.abs(a.get_forces() - a.arrays["student_forces"]); n = len(a)
    dE = (a.info["student_energy"] - a.get_potential_energy()) / n
    fmae.append(dF.mean()); rows[a.info["category"]].append((1000 * dE, 1000 * dF.mean(), 1000 * np.abs(a.get_forces()).mean(), a.info.get("min_d", np.nan)))
    for e in set(a.get_chemical_symbols()):
        m = np.array(a.get_chemical_symbols()) == e; by_el[e].append(1000 * dF[m].mean())
print(f"{'run':28s} {'n':>4s} {'dE/atom meV mean (|max|)':>26s} {'F MAE meV/A':>12s} {'|F| meV/A':>10s} {'min d':>6s}")
for c, v in sorted(rows.items()):
    v = np.array(v)
    print(f"{c:28s} {len(v):4d} {v[:, 0].mean():+10.2f} ({np.abs(v[:, 0]).max():7.2f}) {v[:, 1].mean():12.1f} {v[:, 2].mean():10.0f} {np.nanmin(v[:, 3]):6.2f}")
print("per element F MAE (meV/A):", {e: round(float(np.mean(v)), 1) for e, v in sorted(by_el.items())})
fmae = np.array(fmae); rng = np.random.default_rng(a_.seed)
n_top = int(a_.top * len(fr)); order = np.argsort(-fmae); top = set(order[:n_top].tolist())
rest = [i for i in range(len(fr)) if i not in top]; rnd = set(rng.choice(rest, min(len(rest), int(a_.rand * len(fr))), replace=False).tolist())
sel = sorted(top | rnd)
out = []
for i in sel:
    b = fr[i].copy(); b.calc = fr[i].calc; b.arrays.pop("student_forces", None); b.info.pop("student_energy", None); out.append(b)
write(a_.out, out)
print(f"selected {len(out)} of {len(fr)} (top {n_top} by F MAE >= {1000 * fmae[order[n_top - 1]]:.0f} meV/A, + {len(rnd)} random)")
