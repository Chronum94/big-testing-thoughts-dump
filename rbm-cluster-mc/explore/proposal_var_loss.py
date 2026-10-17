"""Round-2 #5: log-variance loss on data + fresh RBM proposals (vs data-only Var, §4).

Loss: Var_{x ~ r}[d(x)], d = -beta E + F, with r = 50/50 mixture of a data batch and one-step
Gibbs proposals s' ~ K(.|data) regenerated from the current RBM every batch (no gradient through
the sampling). Gradient 2 Cov_r(d, dF/dtheta) as in var_loss.py. On the proposal half this is the
log-variance / reverse-KL-like term (Richter et al. arXiv:2010.10436) that penalises regions where
p_RBM >> pi, which data-only Var (§4) never sees.
Acceptance is measured with several chains (sticky-chain pathology, §3).
"""
import copy
import sys
import warnings
import numpy as np
sys.path.insert(0, '.')
from model import Lattice, generate_dataset
from rbm import RBM, sigmoid, log_ratio, rbm_mh_step
sys.path.insert(0, 'explore')
from var_loss import var_fit


def var_fit_mix(rbm, lat, beta, data, epochs, lr, batch=64, rng=None):
    rng = np.random.default_rng(rng)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(data)), max(1, len(data) // batch)):
            x = np.concatenate([data[idx], rbm.gibbs(data[idx], rng)])
            onehot, ph = np.eye(rbm.q)[x], sigmoid(rbm.hidden_field(x))
            d = -beta * lat.energy(x) + rbm.free_energy(x)
            w = 2 * (d - d.mean()) / len(x)                  # dVar/dF per sample
            rbm.W += lr * np.einsum("b,bia,bm->iam", w, onehot, ph) * rbm.mask
            rbm.b += lr * np.einsum("b,bia->ia", w, onehot)
            rbm.c += lr * (w @ ph)
    return rbm


def moved(rbm, lat, beta, starts, rng, block, n=250):
    """Moved fraction pooled over several independent chains (one per start state)."""
    mv = 0
    for s in starts:
        s = s.copy()
        for _ in range(n):
            sites = None if block < 0 else np.flatnonzero(lat.distance(np.arange(lat.N), rng.integers(lat.N)) <= block)
            s, _, c = rbm_mh_step(rbm, lat, beta, s, rng, sites)
            mv += c > 0
    return mv / (n * len(starts))


def main(L, q, T, seeds, lr=0.001, epochs=30):
    beta, rows = 1 / T, {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        lat = Lattice(L, 2, q=q)
        data = generate_dataset(lat, beta, 2000, thin=5, rng=rng)
        tr, te = data[:1500], data[1500:]
        pcd = RBM(lat.N, q, lat.N, rng=seed).fit(tr, epochs=30, lr=0.05, k=5, rng=seed)
        arms = {"PCD": pcd,
                "PCD+Var(data)": var_fit(copy.deepcopy(pcd), lat, beta, tr, epochs, lr, rng=seed),
                "PCD+Var(data+prop)": var_fit_mix(copy.deepcopy(pcd), lat, beta, tr, epochs, lr, rng=seed)}
        starts = te[rng.choice(len(te), 8, replace=False)]
        for name, rbm in arms.items():
            props = rbm.gibbs(te, rng)
            dd, dp = log_ratio(rbm, lat, beta, te), log_ratio(rbm, lat, beta, props)
            r = dict(sd_data=dd.std(), sd_prop=dp.std(), dshift=dp.mean() - dd.mean(),
                     mv_full=moved(rbm, lat, beta, starts, rng, -1), mv_blk2=moved(rbm, lat, beta, starts, rng, 2))
            rows.setdefault(name, []).append(r)
            print(seed, name, {k: round(float(v), 3) for k, v in r.items()}, flush=True)
    print(f"\n2D L={L} q={q} T={T}: mean ± std over seeds {list(seeds)}")
    for name, rs in rows.items():
        print(f"{name:20s} " + "  ".join(f"{k}={np.mean([r[k] for r in rs]):.3f}±{np.std([r[k] for r in rs]):.3f}" for k in rs[0]))


def tau_check(seed=0, n=4000, chains=6):
    """Full-system RBM move per step vs the work-matched null (one extra Metropolis sweep, §8b)."""
    from model import metropolis_sweep
    from run import tau_int
    L, T = 8, 2.5
    beta, rng = 1 / T, np.random.default_rng(seed)
    lat = Lattice(L, 2)
    data = generate_dataset(lat, beta, 2000, thin=5, rng=rng)
    pcd = RBM(lat.N, 2, lat.N, rng=seed).fit(data[:1500], epochs=30, lr=0.05, k=5, rng=seed)
    mix = var_fit_mix(copy.deepcopy(pcd), lat, beta, data[:1500], 30, 0.001, rng=seed)
    extras = {"single-flip": None, "+extra sweep": "sweep",
              "+PCD full move": pcd, "+Var(data+prop) full move": mix}
    for name, ex in extras.items():
        tE, tm = [], []
        for c in range(chains):
            r = np.random.default_rng(100 + c)
            s = data[1500 + 50 * c].copy()
            E, m = np.empty(n), np.empty(n)
            for t in range(n):
                metropolis_sweep(lat, s, beta, r)
                if ex == "sweep":
                    metropolis_sweep(lat, s, beta, r)
                elif ex is not None:
                    s = rbm_mh_step(ex, lat, beta, s, r)[0]
                E[t], m[t] = lat.energy(s), lat.order_parameter(s)[0]
            tE.append(tau_int(E)); tm.append(tau_int(m))
        print(f"{name:28s} tau_E={np.mean(tE):5.2f}±{np.std(tE)/np.sqrt(chains):.2f}  "
              f"tau_m={np.mean(tm):5.2f}±{np.std(tm)/np.sqrt(chains):.2f}", flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    if sys.argv[1:] == ["tau"]:
        tau_check()
    elif (sys.argv[1] if len(sys.argv) > 1 else "ising") == "ising":
        main(8, 2, 2.5, range(3))
    else:
        main(6, 3, 1.0, range(3))
