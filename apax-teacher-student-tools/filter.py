"""Drop frames by max|F| and an optional info-range. usage: python filter.py IN OUT --fmax 20 [--key rho --max 4.6]"""
import argparse
from collections import Counter

import numpy as np
from ase.io import read, write

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--fmax", type=float, default=1e9); ap.add_argument("--key"); ap.add_argument("--max", type=float, default=np.inf)
a = ap.parse_args()
f = read(a.inp, ":")
keep = [x for x in f if np.abs(x.get_forces()).max() <= a.fmax and (a.key is None or x.info.get(a.key, -np.inf) <= a.max)]
write(a.out, keep)
lost = Counter(x.info["category"] for x in f) - Counter(x.info["category"] for x in keep)
print(f"{a.inp}: kept {len(keep)} of {len(f)}; dropped by category {dict(lost)}")
