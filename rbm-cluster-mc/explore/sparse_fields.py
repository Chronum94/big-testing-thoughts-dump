"""Sparse r=2 RBM on 2D Ising L=8: mean spin-flip weight u_m vs distance from the hidden unit's
site, and fraction of units whose u_m has a single sign (= local magnetisation 'blob' detector)."""
import sys
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, generate_dataset
from rbm import RBM, local_mask

L, r = 8, 2
for T in (1.8, 2.269, 3.5):
    rng = np.random.default_rng(0)
    lat = Lattice(L, 2)
    data = generate_dataset(lat, 1 / T, 1500, thin=5, rng=rng)
    rbm = RBM(lat.N, 2, lat.N, mask=local_mask(lat, 1, r), rng=rng).fit(data, epochs=40, lr=0.05, k=5, rng=rng)
    U = rbm.W[:, 1, :] - rbm.W[:, 0, :]
    dist = lat.distance(np.arange(lat.N)[:, None], np.arange(lat.N)[None, :])
    sgn = np.sign(U[np.arange(lat.N), np.arange(lat.N)])          # orient by centre weight
    prof = [np.mean((U * sgn)[dist == k]) for k in range(r + 1)]
    one_sign = np.mean([np.all(U[dist[:, m] <= r, m] * sgn[m] > 0) for m in range(lat.N)])
    print(f"T={T:5.3f} mean oriented u by distance {np.round(prof, 2)}  single-sign units={one_sign:.2f}", flush=True)
