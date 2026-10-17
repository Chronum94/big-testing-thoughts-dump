"""Smoke test: trained dense vs sparse RBM; acceptance of full vs local-block proposals vs L."""
import sys
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, generate_dataset
from rbm import RBM, local_mask, log_ratio, rbm_mh_step

T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
beta = 1 / T
for L in (4, 6, 8, 12):
    rng = np.random.default_rng(L)
    lat = Lattice(L, 2)
    data = generate_dataset(lat, beta, 2000, thin=5, rng=rng)
    tr, te = data[:1500], data[1500:]
    for name, mask in [("dense", None), ("local r=2", local_mask(lat, 1, 2))]:
        rbm = RBM(lat.N, 2, lat.N, mask=mask, rng=rng).fit(tr, epochs=30, lr=0.05, k=5, rng=rng)
        row = [f"L={L:2d} {name:9s} std d={log_ratio(rbm, lat, beta, te).std():6.2f}"]
        for block in (-1, 1, 2):
            s, acc, sz = te[0].copy(), 0, 0
            for t in range(2000):
                sites = None if block < 0 else np.flatnonzero(lat.distance(np.arange(lat.N), rng.integers(lat.N)) <= block)
                s, a, n = rbm_mh_step(rbm, lat, beta, s, rng, sites)
                acc += n > 0; sz += n
            row.append(f"blk{block:2d}: moved={acc/2000:.3f} |C|={sz/max(acc,1):5.1f}")
        print("  ".join(row), flush=True)
