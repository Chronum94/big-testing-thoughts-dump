"""Pipeline: single-flip MC dataset -> train RBM -> compare single-flip vs RBM-assisted sampling.

    python run.py --d 2 --L 8 --T 2.4                     # 2D ferromagnetic Ising near T_c
    python run.py --d 2 --L 6 --q 3 --T 1.0               # 3-state Potts (T_c = 0.995)
    python run.py --d 2 --L 6 --bonds pm --T 1.0           # +-J spin glass
    python run.py --d 1 --L 32 --bonds staggered --J2 0.2 # dimerized chain
    python run.py ... --radius 1 --block 2                 # sparse RBM + local block moves
"""
import argparse
import time
import warnings

import numpy as np

from model import Lattice, generate_dataset, metropolis_sweep
from rbm import RBM, local_mask, log_ratio, rbm_mh_step


def tau_int(x, c=5.0):
    """Integrated autocorrelation time with Sokal's automatic window (tau=0.5 for iid data)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] == 0:
        return np.nan
    taus = 0.5 + np.cumsum(acf[1:] / acf[0])  # taus[W-1] = tau(W)
    hits = np.flatnonzero(np.arange(1, n) >= c * taus)
    if not hits.size or n < 50 * taus[hits[0]]:
        warnings.warn(f"tau_int unreliable: run of {n} too short (Sokal window needs n >> 50 tau)")
    return taus[hits[0]] if hits.size else np.nan


def run_chain(lat, beta, s, n_steps, rng, rbm=None, n_rbm=0, block=-1):
    """Each step: one single-flip sweep, then n_rbm RBM proposals. Returns observables + stats."""
    s = s.copy()
    E, m, moved, sizes = [], [], 0, []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        metropolis_sweep(lat, s, beta, rng)
        for _ in range(n_rbm):
            sites = None if block < 0 else np.flatnonzero(
                lat.distance(np.arange(lat.N), rng.integers(lat.N)) <= block)
            s, a, n_changed = rbm_mh_step(rbm, lat, beta, s, rng, sites)
            if n_changed:  # accepted s' == s is a no-op; don't count it as a move
                moved += 1
                sizes.append(n_changed)
        E.append(lat.energy(s))
        m.append(lat.order_parameter(s)[0])
    sec = (time.perf_counter() - t0) / n_steps
    return dict(E=np.array(E) / lat.N, m=np.array(m), sec_per_step=sec,
                moved=moved / max(1, n_steps * n_rbm), cluster=np.mean(sizes) if sizes else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--T", type=float, default=2.4)
    ap.add_argument("--bonds", default="uniform", help="uniform | pm | gauss | dilute | staggered")
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--J2", type=float, default=0.5, help="second coupling for staggered bonds")
    ap.add_argument("--p", type=float, default=0.5, help="probability for pm / dilute bonds")
    ap.add_argument("--n-data", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--hidden", type=float, default=1.0, help="hidden units per site")
    ap.add_argument("--radius", type=int, default=-1, help="sparse RBM radius (-1 = dense)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=5, help="Gibbs steps per PCD update")
    ap.add_argument("--n-rbm", type=int, default=1, help="RBM proposals per step")
    ap.add_argument("--block", type=int, default=-1, help="local block radius (-1 = all sites)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    beta = 1.0 / a.T
    lat = Lattice(a.L, a.d, q=a.q, kind=a.bonds, J=a.J, J2=a.J2, p=a.p, rng=rng)
    print(f"{a.d}D L={a.L} q={a.q} bonds={a.bonds} T={a.T}  N={lat.N}")

    t0 = time.perf_counter()
    data = generate_dataset(lat, beta, 2 * a.n_data, thin=a.thin, rng=rng)
    train, test = data[: a.n_data], data[a.n_data:]
    print(f"dataset: {len(data)} samples in {time.perf_counter() - t0:.1f}s; "
          f"tau_int(E) of dataset = {tau_int(lat.energy(data)):.1f} samples (want ~0.5)")

    n_per_site = max(1, round(a.hidden))
    M = n_per_site * lat.N if a.radius >= 0 else max(1, round(a.hidden * lat.N))
    mask = local_mask(lat, n_per_site, a.radius) if a.radius >= 0 else None
    rbm = RBM(lat.N, a.q, M, mask=mask, rng=rng)

    def report(epoch, rbm):
        print(f"  epoch {epoch:3d}  std d(s) on held-out data = "
              f"{log_ratio(rbm, lat, beta, test).std():.3f}  (0 = exact RBM)")
    rbm.fit(train, epochs=a.epochs, lr=a.lr, k=a.k, rng=rng, log_every=max(1, a.epochs // 5),
            callback=report)

    s0 = test[-1]
    print(f"\n{'sampler':>14} {'moved':>6} {'cluster':>8} {'tau_E':>7} {'tau_m':>7} {'ms/step':>8}")
    for name, kw in [("single-flip", {}), ("+RBM", dict(rbm=rbm, n_rbm=a.n_rbm, block=a.block))]:
        r = run_chain(lat, beta, s0, a.steps, rng, **kw)
        print(f"{name:>14} {r['moved']:6.3f} {r['cluster']:8.1f} {tau_int(r['E']):7.1f} "
              f"{tau_int(r['m']):7.1f} {1e3 * r['sec_per_step']:8.2f}   <E>/N = {r['E'].mean():.4f}")
    print("moved = fraction of RBM proposals accepted AND changing >=1 site; cluster = mean #sites changed.")
    print("tau in steps (1 step = 1 sweep [+ n_rbm RBM moves]); compare tau x ms/step for cost.")


if __name__ == "__main__":
    main()
