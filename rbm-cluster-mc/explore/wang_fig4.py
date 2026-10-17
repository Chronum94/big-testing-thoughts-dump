"""Reproduce Wang 2017 Fig. 4 at small L: plaquette Ising (Eq. 6), K/J=0.2, T=2.4955 (paper's T_c).
(a) Binder ratio <M^4>/<M^2>^2 from the BM cluster update (paper: -> 1.1679 for 2D Ising class).
(b) tau_E per sweep: local Metropolis vs BM cluster update. Also ms/step and the fraction of
    antiferromagnetic effective links (beta J - W < 0 is possible; not discussed in the paper)."""
import sys
import time
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice
from bm3 import plaquette_bm_step, plaquette_energy, plaquette_metropolis_sweep, spin
from run import tau_int

warnings.simplefilter("ignore")
K, T = 0.2, 2.4955
beta = 1 / T
print(f"W = arccosh(e^(2 beta K)) = {np.arccosh(np.exp(2 * beta * K)):.4f}, beta J = {beta:.4f}")
for L, n_cl, n_loc in ((8, 20000, 20000), (16, 20000, 20000), (32, 10000, 4000)):
    lat = Lattice(L, 2)
    rng = np.random.default_rng(L)
    out = []
    for name in ("local", "cluster"):
        n = n_loc if name == "local" else n_cl
        s = rng.integers(2, size=lat.N)
        E, M, af = np.empty(n), np.empty(n), []
        for t in range(n // 10 + n):
            if name == "local":
                plaquette_metropolis_sweep(lat, s, beta, K, rng)
            else:
                s, Keff = plaquette_bm_step(lat, beta, K, s, rng)
                af.append((Keff < 0).mean())
            if t == n // 10:
                t0 = time.perf_counter()
            if t >= n // 10:
                E[t - n // 10], M[t - n // 10] = plaquette_energy(lat, s, K), spin(s).sum()
        ms = 1e3 * (time.perf_counter() - t0) / n
        binder = np.mean(M**4) / np.mean(M**2) ** 2
        out.append(f"{name}: tau_E={tau_int(E):7.1f} ms={ms:6.2f} binder={binder:.3f}"
                   + (f" AF-links={np.mean(af):.3f}" if af else ""))
    print(f"L={L:2d}  " + "   ".join(out), flush=True)
