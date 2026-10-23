"""Energy-difference metrics for one model: carbonyl<->ether swap pairs and within-family scan differences.

usage: python eval_pairs.py MODEL_DIR   (CPU; reads data/test_swapall, test_swapemc, test_gen)
Swap pairs (info pair_id, member 0/1): dE = E(swapped) - E(original), total energy in meV (same atoms, so per-element
references cancel). Reports MAE of dE, mean signed error, RMSE, sign agreement (fraction of pairs where the predicted ordering
matches the teacher), Kendall tau over pairs, and the teacher's mean |dE| for scale.
Scan families (test_gen; frames sharing src_frame = same source MD frame at different strains): all within-family pair
differences of per-atom energy, MAE in meV/atom.
"""
import collections
import itertools
import os
import sys

sys.path.insert(0, "/home/chronum/scratch/common"); import cpuload  # noqa: E401,F401
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from apax.md import ASECalculator  # noqa: E402

calc = ASECalculator(sys.argv[1])
for name in ("swapall", "swapemc"):
    f = f"data/test_{name}.extxyz"
    if not os.path.exists(f):
        continue
    fr = read(f, ":"); pr = calc.batch_eval(fr, batch_size=8, silent=True); pairs = collections.defaultdict(dict)
    for a, p in zip(fr, pr):
        pairs[a.info["pair_id"]][a.info["member"]] = (a.get_potential_energy(), p.get_potential_energy(), a.info.get("solvent"), a.info.get("swap"))
    r, q, lab = [], [], []
    for d in pairs.values():
        if 0 in d and 1 in d:
            r.append(1e3 * (d[1][0] - d[0][0])); q.append(1e3 * (d[1][1] - d[0][1])); lab.append(f"{d[0][2]} {d[0][3]}")
    r, q = np.array(r), np.array(q); e = q - r
    print(f"{name}: n {len(r)} pairs | dE MAE {np.abs(e).mean():.1f} meV | mean {e.mean():+.1f} | RMSE {np.sqrt((e ** 2).mean()):.1f} | "
          f"sign agree {np.mean(np.sign(q) == np.sign(r)):.3f} | tau {kendalltau(r, q)[0]:.3f} | teacher mean |dE| {np.abs(r).mean():.1f} meV")
    for k in sorted(set(lab)):
        s = np.array([x == k for x in lab])
        if s.sum() >= 5:
            print(f"  {k}: n {s.sum()} | dE MAE {np.abs(e[s]).mean():.1f} | sign agree {np.mean(np.sign(q[s]) == np.sign(r[s])):.3f}")
fr = read("data/test_gen.extxyz", ":"); pr = calc.batch_eval(fr, batch_size=8, silent=True); fam = collections.defaultdict(list)
for a, p in zip(fr, pr):
    fam[a.info["src_frame"]].append((a.get_potential_energy() / len(a), p.get_potential_energy() / len(a)))
errs = [1e3 * ((qj - qi) - (rj - ri)) for v in fam.values() for (ri, qi), (rj, qj) in itertools.combinations(v, 2)]
print(f"scanfam (test_gen): n {len(errs)} pairs | within-family d(E/atom) MAE {np.mean(np.abs(errs)):.3f} meV/atom | RMSE {np.sqrt(np.mean(np.square(errs))):.3f}")
