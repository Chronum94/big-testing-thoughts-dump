"""MAD-1.6 mc3d + mc3d_rattled + mc3d_random (bulk, periodic), uniform random N from the pooled set (seed 0)."""
import io, random, sys
from collections import Counter
import numpy as np
from ase.io import read, write

n_keep = int(sys.argv[1])
subsets = ["mc3d", "mc3d_rattled", "mc3d_random"]
frames, tags = [], []
with open("mad-1.6-r2scan-train.xyz") as f:
    while (line := f.readline()):
        n = int(line); hdr = f.readline(); body = [f.readline() for _ in range(n)]
        for s in subsets:
            if f"subset={s} " in hdr:
                frames.append(line + hdr + "".join(body)); tags.append(s)
pick = sorted(random.Random(0).sample(range(len(frames)), n_keep))
atoms = read(io.StringIO("".join(frames[i] for i in pick)), index=":", format="extxyz")
write(f"mad_bulk_{n_keep}.traj", atoms)
print("pool", Counter(tags), "kept", Counter(tags[i] for i in pick))
nat = np.array([len(a) for a in atoms]); F = np.concatenate([np.linalg.norm(a.get_forces(), axis=1) for a in atoms])
print(f"{len({z for a in atoms for z in a.numbers})} elements; atoms/frame p50/max {np.median(nat):.0f}/{nat.max()}; "
      f"pbc all {all(a.pbc.all() for a in atoms)}; stress {sum('stress' in a.calc.results for a in atoms)}; "
      f"|F| p50/p90/max {np.median(F):.2f}/{np.percentile(F, 90):.2f}/{F.max():.1f}")
