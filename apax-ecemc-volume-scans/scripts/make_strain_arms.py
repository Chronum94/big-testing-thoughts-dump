"""ortho/gen arms at other strain norms (o03, g03, o12, g12): base s1 MD + anisotropic rigid scans from <arm>_s1/,
subsampled to the isotropic scan arm's per-scale-factor counts (dose-matched). Test sets: data/test_<name>.extxyz from <name>_s2/."""
import glob
import os
import sys
import numpy as np
from ase.io import read, write
rd = lambda pat: [a for f in sorted(glob.glob(pat)) for a in read(f, ":", format="extxyz")]  # noqa: E731
base = rd("md/*.extxyz"); sc = [a for a in read("data/scan/train.extxyz", ":") if "scan_s" in a.info]
for arm in sys.argv[1].split(","):
    an, rng, extra = rd(f"{arm}_s1/labelled_*.extxyz"), np.random.default_rng(0), []
    for s in sorted({round(a.info["scan_s"], 3) for a in sc}):
        need = sum(round(a.info["scan_s"], 3) == s for a in sc); pool = [a for a in an if round(a.info["scan_s"], 3) == s]
        idx = rng.choice(len(pool), min(need, len(pool)), replace=False)  # contact rejection can leave a pool a few frames short:
        idx = np.concatenate([idx, rng.choice(len(pool), need - len(idx))]) if len(idx) < need else idx  # top up with repeats, keep the dose
        if len(pool) < need:
            print(f"  {arm} s={s}: pool {len(pool)} < {need}, {need - len(pool)} repeats")
        extra += [pool[i] for i in idx]
    os.makedirs(f"data/{arm}", exist_ok=True); write(f"data/{arm}/train.extxyz", base + extra); print(f"{arm}: {len(base)} + {len(extra)}")
for t in sys.argv[2].split(","):
    fr = rd(f"{t}/labelled_*.extxyz"); write(f"data/test_{t.replace('_s2', '').replace('swap_', 'swap')}.extxyz", fr); print(f"test {t}: {len(fr)}")
