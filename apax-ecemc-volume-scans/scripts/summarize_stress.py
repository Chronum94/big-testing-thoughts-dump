"""Seed-averaged stress-test tables from logs_eval/ (eval_model.sh output).

usage: python summarize_stress.py [ARM_GROUP ...]   (group = model name without arm_ and _sN; default: every group found)
Per group and test set: all-atom trans / rot / intra force RMSE (meV/A) and energy MAE (meV/atom), mean over seeds with
the seed count; pair metrics: swap dE MAE (meV, total energy), Kendall tau, within-scan-family dE MAE (meV/atom).
Also prints paired ortho-vs-gen differences (same seed) per strain norm.
"""
import collections
import glob
import os
import re
import sys

import numpy as np

SETS = ["md", "gen", "ortho", "scan", "shear", "uniax", "o12", "g12", "o20", "g20", "shift", "x2"]
ROW = re.compile(r"^\| all \| \d+ \| ([\d.]+) \([\d.]+\) \| ([\d.]+) \([\d.]+\) \| ([\d.]+)", re.M)
MDROW = re.compile(r"^MD frames: .*?MAE ([\d.]+) meV/atom \(n=(\d+)\)", re.M)
EROW = re.compile(r"^\| [\d.]+ \| (\d+) \| [-\d.]+ \| ([\d.]+) \|$", re.M)


def split(a):
    m = re.match(r"(.*?)(?:_s(\d))?$", a); return m[1], int(m[2] or 1)


def forces(f):
    s = open(f).read(); m = ROW.search(s)
    if not m:
        return None
    e = [(int(n), float(v)) for n, v in EROW.findall(s)] + [(int(n), float(v)) for v, n in MDROW.findall(s)]
    return [float(x) for x in m.groups()] + [sum(n * v for n, v in e) / sum(n for n, _ in e) if e else np.nan]


def pairs(f):
    s = open(f).read(); g = lambda p: float(re.search(p, s)[1]) if re.search(p, s) else np.nan  # noqa: E731
    return [g(r"swapall:.*?dE MAE ([\d.]+)"), g(r"swapall:.*?tau ([-\d.]+)"), g(r"swapemc:.*?dE MAE ([\d.]+)"),
            g(r"swapemc:.*?tau ([-\d.]+)"), g(r"scanfam.*?MAE ([\d.]+)")]


res = collections.defaultdict(lambda: collections.defaultdict(dict))  # group -> set -> seed -> metrics
for f in glob.glob("logs_eval/*.txt"):
    stem = f[len("logs_eval/"):-4]; a, t = stem.rsplit("_", 1)
    if t not in SETS + ["pairs"]:
        continue
    grp, seed = split(a); v = pairs(f) if t == "pairs" else forces(f)
    if os.environ.get("SEEDS") and str(seed) not in os.environ["SEEDS"].split(","):
        continue
    if v is not None:
        res[grp][t][seed] = v
groups = sys.argv[1:] or sorted(res)


def mean(grp, t):
    d = res[grp].get(t, {}); return (np.nanmean(list(d.values()), 0), len(d)) if d else (None, 0)


print("## forces trans / rot / intra (meV/A), E MAE (meV/atom); n = seeds\n")
for t in SETS:
    rows = [(g, *mean(g, t)) for g in groups if mean(g, t)[1]]
    if not rows:
        continue
    print(f"### test_{t}\n| arm | n | trans | rot | intra | E |\n|---|---|---|---|---|---|")
    for g, m, n in rows:
        print(f"| {g} | {n} | {m[0]:.1f} | {m[1]:.1f} | {m[2]:.1f} | {m[3]:.2f} |")
    print()
print("## pairs: swap dE MAE (meV) / tau, scan-family dE MAE (meV/atom)\n| arm | n | swapall MAE | tau | swapemc MAE | tau | scanfam |\n|---|---|---|---|---|---|---|")
for g in groups:
    m, n = mean(g, "pairs")
    if n:
        print(f"| {g} | {n} | {m[0]:.0f} | {m[1]:.3f} | {m[2]:.0f} | {m[3]:.3f} | {m[4]:.3f} |")
print("\n## paired gen - ortho (same seed), mean ± sd over seeds: trans / rot / intra / E\n| norm | set | n | d trans | d rot | d intra | d E |\n|---|---|---|---|---|---|---|")
for o, g, lab in (("o03", "g03", "0.03"), ("ortho", "gen", "0.06"), ("o12", "g12", "0.12"), ("o20", "g20", "0.20")):
    for t in SETS:
        s = sorted(set(res[o].get(t, {})) & set(res[g].get(t, {})))
        if len(s) >= 2:
            d = np.array([np.subtract(res[g][t][k], res[o][t][k]) for k in s])
            print(f"| {lab} | {t} | {len(s)} | " + " | ".join(f"{d[:, i].mean():+.2f} ± {d[:, i].std(ddof=1):.2f}" for i in range(4)) + " |")
