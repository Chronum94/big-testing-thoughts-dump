"""Follow-up to z_compare: dynamic exponent of exact bond-RBM Gibbs at T_c, more L, 3 seeds.
Runs n = max(40000, 200*tau_guess) updates. Prints tau_|m| mean +- sem over seeds and local slope z."""
import sys
import numpy as np
sys.path.insert(0, '.')
from model import Lattice
from rbm import exact_ising_rbm
from run import tau_int

beta = 1 / 2.269185
Ls, taus = (8, 12, 16, 20, 24), []  # ponytail: dense W is O(N^2); sparse W for larger L
for L in Ls:
    lat = Lattice(L, 2)
    rbm = exact_ising_rbm(lat, beta)
    n = max(40000, int(200 * 0.6 * L**2))
    t = []
    for seed in range(3):
        rng = np.random.default_rng(100 * L + seed)
        s = rng.integers(2, size=lat.N)
        for _ in range(n // 10):
            s = rbm.gibbs(s, rng)
        m = np.empty(n)
        for i in range(n):
            s = rbm.gibbs(s, rng)
            m[i] = abs((2 * s - 1).mean())
        t.append(tau_int(m))
    taus.append(np.mean(t))
    z = "" if len(taus) < 2 else f"  local z={np.log(taus[-1]/taus[-2])/np.log(L/Ls[len(taus)-2]):.2f}"
    print(f"L={L:3d} n={n} tau_|m|={np.mean(t):8.1f} +- {np.std(t)/np.sqrt(3):6.1f}{z}", flush=True)
