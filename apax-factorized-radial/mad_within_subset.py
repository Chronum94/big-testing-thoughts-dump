"""MAD-1.6 dimers + trimers + trimers-extended with all pair distances <= DMAX, random N (seed 0)."""
import io, random, sys
import numpy as np
from ase.io import read, write

n_keep, dmax = int(sys.argv[1]), float(sys.argv[2])
frames = []
with open("mad-1.6-r2scan-train.xyz") as f:
    while (line := f.readline()):
        n = int(line); hdr = f.readline(); body = [f.readline() for _ in range(n)]
        if any(f"subset={s} " in hdr for s in ["dimers", "trimers", "trimers-extended"]):
            frames.append(line + hdr + "".join(body))
atoms = read(io.StringIO("".join(frames)), index=":", format="extxyz")
keep = [a for a in atoms if a.get_all_distances().max() <= dmax]
print(f"pool {len(atoms)}; all d <= {dmax}: {len(keep)} ({100*len(keep)/len(atoms):.0f}%)")
random.Random(0).shuffle(keep); keep = keep[:n_keep]
write(f"mad_dtte_within{int(dmax)}_{n_keep}.traj", keep)
F = np.concatenate([np.linalg.norm(a.get_forces(), axis=1) for a in keep])
print(f"kept {len(keep)}: {sum(len(a)==2 for a in keep)} dimers, {sum(len(a)==3 for a in keep)} trimers, "
      f"{len({z for a in keep for z in a.numbers})} elements; |F| p50/p90/max {np.median(F):.2f}/{np.percentile(F,90):.2f}/{F.max():.1f}")
