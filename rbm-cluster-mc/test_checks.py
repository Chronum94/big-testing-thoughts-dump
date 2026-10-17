"""Correctness checks against exact enumeration on tiny systems. Run: python -m pytest -q"""
import itertools

import numpy as np

from model import Lattice, metropolis_sweep
from rbm import RBM, exact_ising_rbm, log_ratio, rbm_mh_step, sigmoid


def all_states(lat):
    return np.array(list(itertools.product(range(lat.q), repeat=lat.N)))


def exact_probs(lat, beta, states):
    logw = -beta * lat.energy(states)
    p = np.exp(logw - logw.max())
    return p / p.sum()


def rbm_mh_kernel(rbm, lat, beta, sites, accept=True):
    """Exact transition matrix of rbm_mh_step (block = sites), summing over all hidden configs."""
    S = all_states(lat)
    H = np.array(list(itertools.product([0, 1], repeat=rbm.M)), float)
    ph = sigmoid(rbm.hidden_field(S))                                          # (nS, M)
    p_h = np.prod(np.where(H[None] == 1, ph[:, None], 1 - ph[:, None]), -1)   # (nS, nH)
    logits = rbm.b + np.einsum("iam,hm->hia", rbm.W, H)
    logp = logits - np.logaddexp.reduce(logits, -1, keepdims=True)           # (nH, N, q)
    p_block = np.exp(logp[:, sites, S[:, sites]].sum(-1))                     # (nH, nS')
    rest = np.setdiff1d(np.arange(lat.N), sites)
    T = (p_h @ p_block) * (S[:, None, rest] == S[None, :, rest]).all(-1)
    if accept:
        d = log_ratio(rbm, lat, beta, S)
        T *= np.minimum(1, np.exp(d[None, :] - d[:, None]))
    T[np.diag_indices_from(T)] += 1 - T.sum(1)
    return T, exact_probs(lat, beta, S)


def test_free_energy_matches_brute_force_sum_over_hidden():
    rng = np.random.default_rng(0)
    rbm = RBM(N=4, q=3, M=3, scale=1.0, rng=rng)
    rbm.b, rbm.c = rng.normal(size=rbm.b.shape), rng.normal(size=rbm.c.shape)
    for s in rng.integers(3, size=(5, 4)):
        logp = [rbm.b[np.arange(4), s].sum() + h @ rbm.c + rbm.W[np.arange(4), s].sum(0) @ h
                for h in map(np.array, itertools.product([0, 1], repeat=3))]
        assert np.isclose(-rbm.free_energy(s), np.logaddexp.reduce(logp))


def test_exact_ising_rbm_is_exact_for_spin_glass():
    lat = Lattice(3, 2, q=2, kind="gauss", rng=1)
    d = log_ratio(exact_ising_rbm(lat, 0.7), lat, 0.7, all_states(lat))
    assert np.ptp(d) < 1e-9


def test_rbm_mh_detailed_balance_full_and_block():
    """Exact: pi_i T_ij == pi_j T_ji for a random (bad) RBM, full and local-block moves, Ising
    and Potts. Negative control: without the MH step pi is NOT stationary."""
    rng = np.random.default_rng(0)
    cases = [(Lattice(2, 3, q=2, kind="pm", rng=4), np.arange(8)),           # 3D Ising, full
             (Lattice(5, 1, q=3, kind="gauss", rng=6), np.array([1, 2, 3]))]  # 1D Potts, block
    for lat, sites in cases:
        rbm = RBM(lat.N, lat.q, 5, scale=0.8, rng=rng)
        rbm.b = rng.normal(size=rbm.b.shape)
        T, p = rbm_mh_kernel(rbm, lat, 0.6, sites)
        flow = p[:, None] * T
        assert np.abs(flow - flow.T).max() < 1e-14
        T_raw, _ = rbm_mh_kernel(rbm, lat, 0.6, sites, accept=False)
        assert np.abs(p @ T_raw - p).max() > 1e-3


def test_metropolis_samples_boltzmann():
    """Sampling check (the sweep is a Python loop, so test it statistically)."""
    lat, rng, beta = Lattice(5, 1, q=3, kind="pm", rng=2), np.random.default_rng(3), 0.8
    p_exact = exact_probs(lat, beta, all_states(lat))  # itertools order == ravel_multi_index
    s, codes = rng.integers(lat.q, size=lat.N), []
    for _ in range(100_000):
        metropolis_sweep(lat, s, beta, rng)
        codes.append(np.ravel_multi_index(s, (lat.q,) * lat.N))
    p_emp = np.bincount(codes, minlength=len(p_exact)) / len(codes)
    assert 0.5 * np.abs(p_emp - p_exact).sum() < 0.03


def test_tau_int_on_ar1():
    from run import tau_int
    rng, a = np.random.default_rng(0), 0.9
    y = np.zeros(200_000)
    for t, e in enumerate(rng.normal(size=len(y) - 1), 1):
        y[t] = a * y[t - 1] + e
    assert abs(tau_int(y) - 0.5 * (1 + a) / (1 - a)) < 0.5   # exact 9.5
    assert abs(tau_int(rng.normal(size=20_000)) - 0.5) < 0.05


def test_delta_energy_matches_energy_difference():
    rng = np.random.default_rng(0)
    for L, d, q in [(4, 1, 2), (3, 2, 3), (2, 3, 4), (4, 2, 2)]:
        lat = Lattice(L, d, q=q, kind="gauss", rng=rng)
        s = rng.integers(q, size=lat.N)
        for site, new in zip(rng.integers(lat.N, size=20), rng.integers(q, size=20)):
            s2 = s.copy()
            s2[site] = new
            assert np.isclose(lat.delta_energy(s, site, new), lat.energy(s2) - lat.energy(s))


def test_production_rbm_mh_step_preserves_pi():
    """Exercise the real sampler (Gumbel-max, block handling): x ~ pi exactly, one step,
    still ~ pi. Negative control: one raw Gibbs step moves the histogram away from pi."""
    rng = np.random.default_rng(1)
    lat, beta = Lattice(6, 1, q=2, kind="gauss", rng=2), 0.7
    rbm = RBM(lat.N, 2, 4, scale=1.0, rng=rng)
    states = all_states(lat)
    p = exact_probs(lat, beta, states)
    x0 = states[rng.choice(len(p), 30_000, p=p)]
    codes = lambda xs: np.ravel_multi_index(xs.T, (2,) * lat.N)
    tv = lambda xs: 0.5 * np.abs(np.bincount(codes(xs), minlength=len(p)) / len(xs) - p).sum()
    noise = tv(states[rng.choice(len(p), 30_000, p=p)])
    for sites, k in ((None, 1), (np.array([1, 2, 3]), 1), (np.array([1, 2, 3]), 3)):
        x1 = np.array([rbm_mh_step(rbm, lat, beta, x, rng, sites, k)[0] for x in x0])
        assert tv(x1) < 2 * noise + 0.005, (tv(x1), noise)
    assert tv(rbm.gibbs(x0, rng)) > 5 * noise


def test_wolff_samples_boltzmann():
    from model import wolff_step
    lat, rng, beta = Lattice(6, 1, q=2, kind="gauss", rng=3), np.random.default_rng(4), 0.9
    p_exact = exact_probs(lat, beta, all_states(lat))
    s, codes = rng.integers(2, size=lat.N), []
    for _ in range(60_000):
        wolff_step(lat, s, beta, rng)
        codes.append(np.ravel_multi_index(s, (2,) * lat.N))
    p_emp = np.bincount(codes, minlength=len(p_exact)) / len(codes)
    assert 0.5 * np.abs(p_emp - p_exact).sum() < 0.02


def test_gated_lambda_trick_is_exact_for_all_lambda():
    from rbm import gate_strengths, gated_log_ratio
    lat, beta = Lattice(3, 2, q=2, kind="pm", rng=8), 0.8
    for lam in (0.0, 0.3, 1.0):
        rbm, K = exact_ising_rbm(lat, (1 - lam) * beta), gate_strengths(lat, beta, lam)
        assert np.ptp(gated_log_ratio(rbm, lat, beta, K, all_states(lat))) < 1e-9


def test_resample_clusters_matches_brute_force():
    """p(s'|h, n) ∝ prod_i exp(l[i, s'_i]) * prod_{open b} delta_b(s'), by enumeration."""
    from rbm import resample_clusters, satisfied
    rng = np.random.default_rng(9)
    lat = Lattice(5, 1, q=3, kind="gauss", rng=10)
    rbm = RBM(lat.N, 3, 3, scale=1.0, rng=rng)
    S = all_states(lat)
    s = S[rng.integers(len(S))]
    h = rbm.sample_h(s, rng)
    open_ = satisfied(lat, s) & (lat.J > 0)
    assert open_.sum() >= 1
    l = rbm.visible_logits(h)
    w = np.exp(l[np.arange(lat.N), S].sum(-1)) * satisfied(lat, S)[:, open_].all(-1)
    p = w / w.sum()
    codes = [np.ravel_multi_index(resample_clusters(rbm, lat, s, h, open_, rng)[0], (3,) * lat.N)
             for _ in range(20_000)]
    assert 0.5 * np.abs(np.bincount(codes, minlength=len(p)) / len(codes) - p).sum() < 0.03


def test_gated_mh_step_preserves_pi():
    from rbm import gate_strengths, gated_mh_step, resample_clusters, satisfied
    rng = np.random.default_rng(11)
    lat, beta = Lattice(5, 1, q=3, kind="gauss", rng=12), 0.9
    rbm = RBM(lat.N, 3, 3, scale=1.0, rng=rng)
    K = gate_strengths(lat, beta, 0.6)
    S = all_states(lat)
    p = exact_probs(lat, beta, S)
    codes = lambda xs: np.ravel_multi_index(xs.T, (3,) * lat.N)
    tv = lambda xs: 0.5 * np.abs(np.bincount(codes(xs), minlength=len(p)) / len(xs) - p).sum()
    x0 = S[rng.choice(len(p), 30_000, p=p)]
    noise = tv(S[rng.choice(len(p), 30_000, p=p)])
    x1 = np.array([gated_mh_step(rbm, lat, beta, K, x, rng)[0] for x in x0])
    assert tv(x1) < 2 * noise + 0.005, (tv(x1), noise)

    def raw(x):  # negative control: the proposal without MH
        open_ = satisfied(lat, x) & (rng.random(len(K)) < 1 - np.exp(-K))
        return resample_clusters(rbm, lat, x, rbm.sample_h(x, rng), open_, rng)[0]
    assert tv(np.array([raw(x) for x in x0])) > 5 * noise


# --- Wang 2017 three-body BMs (bm3.py) ---------------------------------------------------------

def test_link_bm_eq5_exact_any_b_and_sign():
    from bm3 import link_bm_logp, link_bm_weights
    lat, beta = Lattice(3, 2, q=2, kind="gauss", rng=13), 0.8
    S = all_states(lat)
    for b in (-30.0, -2.0, 0.0, 3.0):
        W = link_bm_weights(lat, beta, b)
        assert np.ptp(link_bm_logp(lat, W, b, S) + beta * lat.energy(S)) < 1e-8
    W = link_bm_weights(lat, beta, -30.0)   # b -> -inf: activation 1 - e^{-2 beta |J|} on satisfied links = SW
    ferro = lat.J > 0
    assert np.allclose(sigmoid(W[ferro] - 30.0), 1 - np.exp(-2 * beta * lat.J[ferro]), atol=1e-6)


def _pi_to_pi(lat, p, step, rng, n=30_000):
    S = all_states(lat)
    codes = lambda xs: np.ravel_multi_index(xs.T, (2,) * lat.N)
    tv = lambda xs: 0.5 * np.abs(np.bincount(codes(xs), minlength=len(p)) / len(xs) - p).sum()
    x0 = S[rng.choice(len(p), n, p=p)]
    return tv(np.array([step(x) for x in x0])), tv(S[rng.choice(len(p), n, p=p)])


def test_link_bm_step_exact_even_with_wrong_weights():
    from bm3 import link_bm_step, link_bm_weights
    rng, lat, beta = np.random.default_rng(14), Lattice(6, 1, q=2, kind="pm", rng=15), 1.0
    p = exact_probs(lat, beta, all_states(lat))
    W = 0.4 * link_bm_weights(lat, beta, 0.5)          # deliberately wrong -> MH must fix it
    tv, noise = _pi_to_pi(lat, p, lambda x: link_bm_step(lat, beta, W, 0.5, x, rng)[0], rng)
    assert tv < 2 * noise + 0.005, (tv, noise)

    def no_mh(x):  # negative control: accept every proposal
        from bm3 import bond_products
        from rbm import clusters
        h = rng.random(len(lat.J)) < sigmoid(W * bond_products(lat, x) + 0.5)
        lab = clusters(lat.N, lat.i[h], lat.j[h])
        return x ^ (rng.random(lab.max() + 1) < 0.5)[lab]
    assert _pi_to_pi(lat, p, no_mh, rng)[0] > 5 * noise


def test_plaquette_bm_step_preserves_plaquette_boltzmann():
    from bm3 import plaquette_bm_step, plaquette_energy
    rng, lat, beta, K = np.random.default_rng(16), Lattice(3, 2, q=2, kind="pm", rng=0), 0.25, 1.0
    S = all_states(lat)
    logw = -beta * plaquette_energy(lat, S, K)
    p = np.exp(logw - logw.max()); p /= p.sum()
    # n=80k: one wrong-K step from pi moves TV only ~0.1, so smaller n lacks power (measured)
    tv, noise = _pi_to_pi(lat, p, lambda x: plaquette_bm_step(lat, beta, K, x, rng)[0], rng, n=80_000)
    assert tv < 1.5 * noise + 0.005, (tv, noise)
    # negative control: the production move run at the wrong K must be detected
    assert _pi_to_pi(lat, p, lambda x: plaquette_bm_step(lat, beta, 0.5, x, rng)[0], rng, n=80_000)[0] > 3 * noise
    # local baseline (plaquette_metropolis_sweep) on the same target
    from bm3 import plaquette_metropolis_sweep

    def sweep(x):
        x = x.copy()
        plaquette_metropolis_sweep(lat, x, beta, K, rng)
        return x
    tv_m, noise_m = _pi_to_pi(lat, p, sweep, rng, n=20_000)
    assert tv_m < 1.5 * noise_m + 0.005, (tv_m, noise_m)


def test_kbd_bm_marginal_exact_ffim_and_pmJ():
    from bm3 import kbd_logp, kbd_weights
    for kind in ("ffim", "pm"):
        lat, beta = Lattice(4, 2, q=2, kind=kind, rng=17), 0.7
        S = all_states(lat)
        for b in (-30.0, -1.0, 0.0, 2.0):
            d = kbd_logp(lat, beta, kbd_weights(lat, beta, b), b, S) + beta * lat.energy(S)
            assert np.ptp(d) < 1e-8, (kind, b, np.ptp(d))


def test_kbd_freezing_conserves_frustrated_plaquette_energy():
    """Eq. A2 lemma: cluster flips keep F_p of every frustrated plaquette that froze a pair."""
    from bm3 import bond_products, checkerboard, kbd_proposal, kbd_weights
    rng = np.random.default_rng(18)
    for kind in ("ffim", "pm"):
        lat, beta, b = Lattice(8, 2, q=2, kind=kind, rng=19), 0.8, 0.5
        cb = checkerboard(lat)
        frus = np.sign(lat.J)[cb].prod(1) < 0
        F = lambda x: (np.sign(lat.J)[cb] * bond_products(lat, x)[cb]).sum(1)
        for _ in range(200):
            s = rng.integers(2, size=lat.N)
            prop, frozen = kbd_proposal(lat, beta, kbd_weights(lat, beta, b), b, s, rng, cb)
            touched = frus & frozen[cb].any(1)
            assert np.array_equal(F(prop)[touched], F(s)[touched])
            assert np.array_equal((lat.J * bond_products(lat, prop) > 0)[frozen],
                                  (lat.J * bond_products(lat, s) > 0)[frozen])


def test_kbd_chain_matches_exact_energy():
    from bm3 import kbd_step, kbd_weights
    from run import tau_int
    rng = np.random.default_rng(20)
    for kind, b in (("ffim", -30.0), ("pm", 0.0)):
        lat, beta = Lattice(4, 2, q=2, kind=kind, rng=21), 0.6
        S = all_states(lat)
        E_exact = exact_probs(lat, beta, S) @ lat.energy(S)
        W, s, E = kbd_weights(lat, beta, b), rng.integers(2, size=lat.N), []
        for _ in range(20_000):
            s = kbd_step(lat, beta, W, b, s, rng)[0]
            E.append(lat.energy(s))
        E = np.array(E)
        err = E.std() * np.sqrt(2 * tau_int(E) / len(E))
        assert abs(E.mean() - E_exact) < 4 * err, (kind, E.mean(), E_exact, err)


def _random_prm(fbm, rng):
    theta = np.concatenate([rng.normal(0, 1, 12), rng.normal(-2, 0.5, 12)])
    prm = fbm.typed(theta)
    prm["Wl"], prm["bl"] = rng.normal(0, 1, len(prm["Wl"])), rng.normal(-2, 0.5, len(prm["bl"]))
    return prm


def test_feature_bm_exact_split_is_exact_and_kbd_matches():
    """exact_split is rejection-free on its whole family; its KBD point equals bm3.kbd_* (same marginal)."""
    from bm3 import FeatureBM, kbd_logp, kbd_weights
    rng = np.random.default_rng(22)
    for kind in ("ffim", "pm"):
        lat, beta = Lattice(4, 2, q=2, kind=kind, rng=26), 0.7
        fbm, S = FeatureBM(lat), all_states(lat)
        for _ in range(4):
            a0 = rng.random(); a1 = rng.random() * (1 - a0)
            prm = fbm.exact_split(lat, beta, a0, a1, *rng.normal(-1, 2, 3))
            assert np.ptp(-beta * lat.energy(S) - fbm.logp(lat, prm, S)) < 1e-8
    lat = Lattice(4, 2, q=2, kind="ffim")
    fbm, S = FeatureBM(lat), all_states(lat)
    ref = kbd_logp(lat, beta, kbd_weights(lat, beta, -30.0), -30.0, S)
    assert np.ptp(fbm.logp(lat, fbm.kbd(lat, beta), S) - ref) < 1e-8


def test_feature_bm_lemma_for_random_params():
    """Frozen bonds keep their status, and a fully frozen parallel pair keeps its feature value."""
    from bm3 import FeatureBM, bond_products
    rng = np.random.default_rng(23)
    for kind in ("ffim", "pm"):
        lat = Lattice(8, 2, q=2, kind=kind, rng=23)
        fbm = FeatureBM(lat)
        for _ in range(100):
            prm, s = _random_prm(fbm, rng), rng.integers(2, size=lat.N)
            prop, frozen = fbm.proposal(lat, prm, s, rng)
            sat = lambda x: lat.J * bond_products(lat, x) > 0
            assert np.array_equal(sat(s)[frozen], sat(prop)[frozen])
            F0, F1 = fbm.features(lat, s)[0], fbm.features(lat, prop)[0]
            assert ((frozen[fbm.bonds[:, :2]].all(1) <= (F0[:, 0] == F1[:, 0])) &
                    (frozen[fbm.bonds[:, 2:]].all(1) <= (F0[:, 1] == F1[:, 1]))).all()


def test_feature_bm_step_preserves_pi_for_random_params():
    """x ~ pi exactly (4x4, enumerated), one feature-BM move, joint (E, M) histogram still ~ pi.
    (A pure-feature chain from a random start mixes too slowly at random params for a chain test.)
    Negative control (FFIM): the raw proposal without MH must distort the histogram."""
    from bm3 import FeatureBM
    rng = np.random.default_rng(24)
    for kind in ("ffim", "pm"):
        lat, beta = Lattice(4, 2, q=2, kind=kind, rng=25), 0.6
        fbm, S = FeatureBM(lat), all_states(lat)
        p = exact_probs(lat, beta, S)
        key = lambda xs: (lat.energy(xs) + 100).astype(int) * 100 + xs.sum(1)     # joint (E, M) level
        levels, inv = np.unique(key(S), return_inverse=True)
        p_EM = np.bincount(inv, weights=p)
        tv = lambda xs: 0.5 * np.abs(np.bincount(np.searchsorted(levels, key(xs)), minlength=len(levels)) / len(xs) - p_EM).sum()
        # sparse activations: dense ones make one giant cluster whose flip is a trivial global Z2 move
        prm = _random_prm(fbm, rng)
        x0 = S[rng.choice(len(p), 20_000, p=p)]
        noise = tv(S[rng.choice(len(p), 20_000, p=p)])
        x1 = np.array([fbm.step(lat, beta, prm, x, rng)[0] for x in x0])
        assert tv(x1) < 2 * noise + 0.005, (kind, tv(x1), noise)
        if kind == "ffim":  # control has power here; for +-J the raw distortion is only ~1.7x noise at this n
            raw = np.array([fbm.proposal(lat, prm, x, rng)[0] for x in x0])
            assert tv(raw) > 5 * noise, (kind, tv(raw), noise)
