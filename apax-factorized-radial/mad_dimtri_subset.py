"""Stream MAD-1.6, keep subset in {dimers, trimers}, sample N frames -> traj."""
import io, random, sys
from collections import Counter
from itertools import combinations_with_replacement
from ase.io import read, write

src, n_keep, seed = "mad-1.6-r2scan-train.xyz", int(sys.argv[1]), 0
frames = []
with open(src) as f:
    while (line := f.readline()):
        n = int(line)
        hdr = f.readline()
        body = [f.readline() for _ in range(n)]
        if "subset=dimers " in hdr or "subset=trimers " in hdr:
            frames.append(line + hdr + "".join(body))
print("dimers+trimers frames:", len(frames))
random.Random(seed).shuffle(frames)
atoms = read(io.StringIO("".join(frames[:n_keep])), index=":", format="extxyz")
write(f"mad_dimtri_{n_keep}.traj", atoms)

pairs = Counter()
for a in atoms:
    Z = sorted(set(a.numbers))
    pairs.update(combinations_with_replacement(Z, 2))
els = sorted({z for a in atoms for z in a.numbers})
print(f"kept {len(atoms)} frames, {sum(len(a) for a in atoms)} atoms, {len(els)} elements, "
      f"{len(pairs)} distinct element pairs; pair count quartiles:",
      sorted(pairs.values())[len(pairs)//4], sorted(pairs.values())[len(pairs)//2],
      sorted(pairs.values())[3*len(pairs)//4], "max", max(pairs.values()))
print("has energy/forces:", atoms[0].get_potential_energy(), atoms[0].get_forces().shape)
