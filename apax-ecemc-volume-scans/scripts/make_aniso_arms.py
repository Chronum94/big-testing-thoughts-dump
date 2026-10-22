"""ortho / gen arms: base s1 MD + labelled anisotropic rigid scans, subsampled to the isotropic scan arm's exact per-scale-factor
counts (dose-matched: same source-frame pool, same s values, same volume change, same strain norm). Also writes
data/test_ortho.extxyz and data/test_gen.extxyz from the s2 anisotropic scans."""
import glob
import os
import numpy as np
from ase.io import read, write
rd = lambda pat: [a for f in sorted(glob.glob(pat)) for a in read(f, ":", format="extxyz")]  # noqa: E731
base = rd("md/*.extxyz"); sc = [a for a in read("data/scan/train.extxyz", ":") if "scan_s" in a.info]
for arm in ("ortho", "gen"):
    an, rng, extra = rd(f"{arm}_s1/labelled_*.extxyz"), np.random.default_rng(0), []
    for s in sorted({round(a.info["scan_s"], 3) for a in sc}):
        need = sum(round(a.info["scan_s"], 3) == s for a in sc); pool = [a for a in an if round(a.info["scan_s"], 3) == s]
        assert len(pool) >= need, (arm, s, len(pool), need)
        extra += [pool[i] for i in rng.choice(len(pool), need, replace=False)]
    os.makedirs(f"data/{arm}", exist_ok=True); write(f"data/{arm}/train.extxyz", base + extra)
    t = rd(f"{arm}_s2/labelled_*.extxyz"); write(f"data/test_{arm}.extxyz", t)
    print(f"{arm}: {len(base)} + {len(extra)}; test_{arm} {len(t)}")
