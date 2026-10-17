"""Round-2 #3: gated moves on the 2D +-J spin glass, L=8. Exact residual RBM -> acceptance 1.
Arms: Metropolis sweeps; uniform lambda; frustration-aware lambda_b (lambda=1 on bonds touching no
frustrated plaquette, lam_f on the rest). Observables: tau_E and mean single-spin tau (glass-relevant).
Cost-fair: tau x ms/step. 2 disorder samples x 2 chains."""
import sys
import time
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, metropolis_sweep
from rbm import exact_ising_rbm, gated_mh_step
from run import tau_int

warnings.simplefilter("ignore")
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
L, beta, n = 8, 1 / T, 6000


def frustrated_bonds(lat):
    """Bonds adjacent to >= 1 frustrated plaquette (2D). Bond index = axis * N + site."""
    N, site = lat.N, np.arange(lat.N).reshape(L, L)
    up = lambda p, ax: np.roll(site, -1, axis=ax).ravel()[p]       # neighbour along axis
    p = np.arange(N)
    bonds = np.stack([p, N + up(p, 0), up(p, 1), N + p], 1)       # x@p, y@p+x, x@p+y, y@p
    frus = np.prod(np.sign(lat.J[bonds]), 1) < 0
    out = np.zeros(len(lat.J), bool)
    out[bonds[frus].ravel()] = True
    return out


def run(lat, step, rng):
    s = rng.integers(2, size=lat.N)
    for _ in range(500):
        s = step(s, rng)
    S, E, t0 = np.empty((n, lat.N)), np.empty(n), time.perf_counter()
    for t in range(n):
        s = step(s, rng)
        S[t], E[t] = s, lat.energy(s)
    ms = 1e3 * (time.perf_counter() - t0) / n
    tau_s = np.nanmean([tau_int(S[:, i]) for i in range(lat.N)])
    return tau_int(E), tau_s, ms


arms = {"metropolis": None}
for lam in (0.0, 0.5, 0.8, 1.0):
    arms[f"uniform lam={lam}"] = ("u", lam)
for lam_f in (0.0, 0.5):
    arms[f"frus-aware lam_f={lam_f}"] = ("f", lam_f)
res = {k: [] for k in arms}
for dis in range(2):
    lat = Lattice(L, 2, q=2, kind="pm", rng=100 + dis)
    fb = frustrated_bonds(lat)
    for name, arm in arms.items():
        if arm is None:
            step = lambda s, rng: (metropolis_sweep(lat, s, beta, rng), s)[1]
        else:
            lam_b = np.full(len(lat.J), arm[1]) if arm[0] == "u" else np.where(fb, arm[1], 1.0)
            lat_res = Lattice(L, 2, q=2, kind="pm", rng=100 + dis)
            lat_res.J = (1 - lam_b) * lat.J
            rbm = exact_ising_rbm(lat_res, beta)
            K = 2 * lam_b * beta * np.abs(lat.J)
            step = lambda s, rng, rbm=rbm, K=K: gated_mh_step(rbm, lat, beta, K, s, rng)[0]
        for c in range(2):
            res[name].append(run(lat, step, np.random.default_rng(10 * dis + c)))
for name, r in res.items():
    r = np.array(r)
    print(f"{name:24s} tau_E={r[:,0].mean():7.1f}  tau_spin={r[:,1].mean():7.1f}  ms/step={r[:,2].mean():.2f}"
          f"  tau_spin*ms={np.mean(r[:,1]*r[:,2]):8.1f}", flush=True)
