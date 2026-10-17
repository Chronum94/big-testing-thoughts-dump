"""Idea #3: k-step Gibbs proposals (still exact). Does the expected accepted jump
<moved * |C|> peak at intermediate k? 2D Ising L=8, T=2.5, trained dense RBM, 2 seeds."""
import sys
import time
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, generate_dataset
from rbm import RBM, rbm_mh_step

beta, L, n = 1 / 2.5, 8, 1500
for seed in (0, 1):
    rng = np.random.default_rng(seed)
    lat = Lattice(L, 2)
    data = generate_dataset(lat, beta, 1500, thin=5, rng=rng)
    rbm = RBM(lat.N, 2, lat.N, rng=rng).fit(data[:1000], epochs=30, lr=0.05, k=5, rng=rng)
    for block in (-1, 2):
        for k in (1, 2, 4, 8, 16, 64):
            s, moved, jump = data[-1].copy(), 0, 0
            t0 = time.perf_counter()
            for _ in range(n):
                sites = None if block < 0 else np.flatnonzero(lat.distance(np.arange(lat.N), rng.integers(lat.N)) <= block)
                s, a, c = rbm_mh_step(rbm, lat, beta, s, rng, sites, k)
                moved += c > 0; jump += c
            ms = 1e3 * (time.perf_counter() - t0) / n
            print(f"seed {seed} block {block:2d} k={k:3d}  moved={moved/n:.3f}  <jump>={jump/n:6.2f} sites/proposal"
                  f"  |C|={jump/max(moved,1):5.1f}  jump/ms={jump/n/ms:6.2f}", flush=True)
