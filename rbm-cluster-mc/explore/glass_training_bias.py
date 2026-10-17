"""Idea #6: in the glassy phase, does training on a non-ergodic single-flip chain leave a large
KL(pi || p_RBM) that the held-out std d(s) diagnostic cannot see (held-out data comes from the
same biased chain)?  2D 4x4 +-J, exact enumeration of all 2^16 states.

Per disorder seed, three RBMs with identical budget (same #samples, epochs, lr, k, M):
  a) single-flip chain samples, augment=False
  b) the same chain samples, augment=True (random global spin flip per sample)
  c) iid exact samples from pi, augment=False
  d) positive control: iid exact samples + EXACT negative phase (enumerate all visibles),
     isolating PCD failure from data bias
"""
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
from model import Lattice, generate_dataset
from rbm import RBM, log_ratio
from run import tau_int

T, L, M, SEEDS = 0.4, 4, 16, range(6)
N_TRAIN, N_TEST, N_THERM = 2000, 2000, 200
FIT = dict(epochs=60, lr=0.05, k=5, batch=64)
beta = 1 / T

STATES = np.array(list(itertools.product(range(2), repeat=L * L)))
CODE = 2 ** np.arange(L * L)[::-1]  # itertools order == binary code
ONEHOT = np.eye(2)[STATES].reshape(len(STATES), -1)  # (2^N, N*q)


def code(s):
    return s @ CODE


def kl_exact(rbm, logpi):
    logp = -rbm.free_energy(STATES)
    logp -= np.logaddexp.reduce(logp)
    pi = np.exp(logpi)
    return float((pi * (logpi - logp)).sum())


def fit_exact_negative(rbm, data, epochs, lr, batch, rng, **_):
    """Same SGD loop as RBM.fit, but <dF/dtheta>_model by exact enumeration instead of PCD."""
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(data)), max(1, len(data) // batch)):
            field = ONEHOT @ rbm.W.reshape(-1, rbm.M) + rbm.c          # (2^N, M)
            logp = ONEHOT @ rbm.b.ravel() + np.logaddexp(0, field).sum(1)  # = -F
            p = np.exp(logp - np.logaddexp.reduce(logp))
            pv = ONEHOT * p[:, None]
            ph = 1 / (1 + np.exp(-field))
            neg = ((pv.T @ ph).reshape(rbm.W.shape), pv.sum(0).reshape(rbm.b.shape), p @ ph)
            pos = rbm._stats(data[idx])
            rbm.W += lr * (pos[0] - neg[0]) * rbm.mask
            rbm.b += lr * (pos[1] - neg[1])
            rbm.c += lr * (pos[2] - neg[2])
    return rbm


def eq_moved(rbm, lat, x, rng):
    """Mean moved fraction of one full-system RBM-MH proposal from starting states x."""
    prop = rbm.gibbs(x, rng)
    a = np.minimum(1, np.exp(log_ratio(rbm, lat, beta, prop) - log_ratio(rbm, lat, beta, x)))
    return float((a * (prop != x).any(1)).mean())


rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    lat = Lattice(L, 2, q=2, kind="pm", rng=seed)
    logw = -beta * lat.energy(STATES)
    logpi = logw - np.logaddexp.reduce(logw)
    pi = np.exp(logpi)

    chain = generate_dataset(lat, beta, N_TRAIN + N_TEST, n_therm=N_THERM, thin=1, rng=rng)
    train_ch, test_ch = chain[:N_TRAIN], chain[N_TRAIN:]
    exact = STATES[rng.choice(len(pi), N_TRAIN + N_TEST, p=pi)]
    train_ex, test_ex = exact[:N_TRAIN], exact[N_TRAIN:]

    # Ergodicity of the chain: pi-mass of visited states, raw and closed under global flip.
    vis = np.unique(code(chain))
    vis_z2 = np.union1d(vis, code(1 - STATES[vis]))
    chain_up = (chain[:, 0] == 1).mean()
    info = dict(seed=seed, mass=pi[vis].sum(), mass_z2=pi[vis_z2].sum(),
                n_eff=1 / (pi ** 2).sum(), up=chain_up, tauE=tau_int(lat.energy(chain)))

    for name, data, aug in [("a raw", train_ch, False), ("b aug", train_ch, True),
                            ("c exact", train_ex, False), ("d exactG", train_ex, None)]:
        r = np.random.default_rng(100 + seed)  # same init + minibatch stream for all
        rbm = RBM(lat.N, 2, M, rng=r)
        if aug is None:
            fit_exact_negative(rbm, data, rng=r, **FIT)
        else:
            rbm.fit(data, augment=aug, rng=r, **FIT)
        rows.append(dict(info, model=name, kl=kl_exact(rbm, logpi),
                         sd_ch=log_ratio(rbm, lat, beta, test_ch).std(),
                         sd_ex=log_ratio(rbm, lat, beta, test_ex).std(),
                         mv_ch=eq_moved(rbm, lat, test_ch, rng),
                         mv_ex=eq_moved(rbm, lat, test_ex, rng)))
        r_ = rows[-1]
        print(f"seed {seed} n_eff={info['n_eff']:6.1f} mass={info['mass']:.3f} "
              f"mass_z2={info['mass_z2']:.3f} up={chain_up:.2f} tauE={info['tauE']:5.1f} | "
              f"{name:8s} KL={r_['kl']:6.3f} sd_d chain={r_['sd_ch']:5.2f} exact={r_['sd_ex']:5.2f} "
              f"moved chain={r_['mv_ch']:.3f} exact={r_['mv_ex']:.3f}", flush=True)

print("\nmean +- sem over seeds (heavy-tailed: see per-seed rows)")
for name in ("a raw", "b aug", "c exact", "d exactG"):
    sel = [r for r in rows if r["model"] == name]
    f = lambda k: f"{np.mean([r[k] for r in sel]):.3f}+-{np.std([r[k] for r in sel]) / np.sqrt(len(sel) - 1):.3f}"
    print(f"{name:8s} KL={f('kl')} sd_chain={f('sd_ch')} sd_exact={f('sd_ex')} "
          f"moved_chain={f('mv_ch')} moved_exact={f('mv_ex')}")
