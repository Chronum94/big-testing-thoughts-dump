"""Student vs teacher elastic constants -> markdown tables (relaxed-ion), plus MAPE summary.
usage: compare.py TEACHER.jsonl STUDENT.jsonl [STUDENT.jsonl ...]
"""
import json
import sys

import numpy as np

KEYS_CUBIC = ("C11", "C12", "C44", "B")


def load(f):
    return {r["name"]: r for r in map(json.loads, open(f))}


T = load(sys.argv[1])
for f in sys.argv[2:]:
    S = load(f)
    names = [n for n in T if n in S]
    if not names:
        continue
    keys = KEYS_CUBIC if "B" in T[names[0]]["relaxed"] else ("C11", "C33", "C12", "C13", "C44", "C66", "B_V", "G_V")
    model = S[names[0]]["model"]
    print(f"\n### {model} vs teacher (relaxed-ion, GPa; student / teacher / Δ%)\n")
    print("| material | a or V Δ% | " + " | ".join(keys) + " | stable s/t |")
    print("|" + "---|" * (len(keys) + 3))
    rel = {k: [] for k in keys}
    for n in names:
        t, s = T[n], S[n]
        geo = (s["a"] / t["a"] - 1) * 100 if "a" in t else (s["vol_per_atom"] / t["vol_per_atom"] - 1) * 100
        cells = []
        for k in keys:
            tv, sv = t["relaxed"][k], s["relaxed"][k]
            d = (sv / tv - 1) * 100 if abs(tv) > 1 else float("nan")
            rel[k].append(abs(d))
            cells.append(f"{sv:.1f} / {tv:.1f} / {d:+.0f}")
        st = lambda r: r["relaxed"].get("stable", r["relaxed"].get("min_eig", 0) > 0)  # noqa: E731
        print(f"| {n} | {geo:+.2f} | " + " | ".join(cells) + f" | {st(s)}/{st(t)} |")
    print("| **median abs Δ%** | | " + " | ".join(f"{np.nanmedian(v):.1f}" for v in rel.values()) + " | |")
