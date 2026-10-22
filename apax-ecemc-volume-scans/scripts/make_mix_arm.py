"""mix arm: base s1 MD + 50/50 ortho and gen anisotropic scans, per-scale-factor counts equal to the scan arm (dose-matched)."""
import glob
import os
import numpy as np
from ase.io import read, write
rd = lambda pat: [a for f in sorted(glob.glob(pat)) for a in read(f, ":", format="extxyz")]  # noqa: E731
base = rd("md/*.extxyz"); sc = [a for a in read("data/scan/train.extxyz", ":") if "scan_s" in a.info]
pools = {arm: rd(f"{arm}_s1/labelled_*.extxyz") for arm in ("ortho", "gen")}; rng = np.random.default_rng(3); extra = []
for s in sorted({round(a.info["scan_s"], 3) for a in sc}):
    need = sum(round(a.info["scan_s"], 3) == s for a in sc); split = {"ortho": need // 2, "gen": need - need // 2}
    for arm, k in split.items():
        pool = [a for a in pools[arm] if round(a.info["scan_s"], 3) == s]
        extra += [pool[i] for i in rng.choice(len(pool), k, replace=False)]
    print(f"s {s}: {split}")
os.makedirs("data/mix", exist_ok=True); write("data/mix/train.extxyz", base + extra); print(f"mix: {len(base)} + {len(extra)}")
