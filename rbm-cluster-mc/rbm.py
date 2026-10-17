"""Restricted Boltzmann machine with categorical (q-state) visible units and binary hidden units,
plus the Metropolis-Hastings move that turns a trained RBM into a (cluster-like) MC proposal.

Visible s_i in {0..q-1} (one-hot internally), hidden h_m in {0, 1}:

    p(s, h) ∝ exp( sum_i b[i, s_i] + sum_m c[m] h_m + sum_{i,m} W[i, s_i, m] h_m )

q = 2 is an ordinary binary RBM. Marginal p(s) ∝ exp(-F(s)) with free energy

    F(s) = -sum_i b[i, s_i] - sum_m softplus(c[m] + sum_i W[i, s_i, m])

The proposal s -> h ~ p(h|s) -> s' ~ p(s'|h) is reversible w.r.t. p_RBM(s), so the exact
acceptance for the target pi(s) ∝ exp(-beta E(s)) only needs F, not the intractable sum over h:

    A(s -> s') = min(1, exp( d(s') - d(s) )),   d(s) = -beta E(s) + F(s) = log pi/p_RBM + const

(Huang & Wang, PRB 95, 035105 (2017), arXiv:1610.02746; L. Wang, PRE 96, 051301 (2017),
arXiv:1702.08586.) The same holds if only a block A of sites takes its new values (see
`rbm_mh_step`): visibles are conditionally independent given h, so updating (h, s_A) at fixed
s_rest is a Gibbs step on p(s, h) and stays reversible w.r.t. p_RBM.
"""
import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow-free


class RBM:
    def __init__(self, N, q, M, mask=None, scale=0.01, rng=None):
        """N sites, q states per site, M hidden units. mask: optional (N, M) bool connectivity."""
        rng = np.random.default_rng(rng)
        self.N, self.q, self.M = N, q, M
        self.mask = np.ones((N, 1, M)) if mask is None else mask[:, None, :].astype(float)
        self.W = scale * rng.normal(size=(N, q, M)) * self.mask
        self.b = np.zeros((N, q))
        self.c = np.zeros(M)

    def hidden_field(self, s):
        """c_m + sum_i W[i, s_i, m];  s: (..., N) ints -> (..., M)."""
        return self.c + self.W[np.arange(self.N), s].sum(-2)

    def free_energy(self, s):
        return (-self.b[np.arange(self.N), s].sum(-1)
                - np.logaddexp(0.0, self.hidden_field(s)).sum(-1))

    def sample_h(self, s, rng):
        p = sigmoid(self.hidden_field(s))
        return (rng.random(p.shape) < p).astype(float)

    def visible_logits(self, h):
        """l[i, a] = b[i, a] + sum_m W[i, a, m] h_m, so that p(s_i = a | h) ∝ exp(l[i, a])."""
        return self.b + np.einsum("iam,...m->...ia", self.W, h)

    def sample_s(self, h, rng):
        logits = self.visible_logits(h)
        gumbel = -np.log(-np.log(rng.random(logits.shape)))  # Gumbel-max = categorical sample
        return np.argmax(logits + gumbel, axis=-1)

    def gibbs(self, s, rng):
        return self.sample_s(self.sample_h(s, rng), rng)

    def _stats(self, s):
        """Gradient of -F(s) w.r.t. (W, b, c), averaged over the batch."""
        onehot = np.eye(self.q)[s]                 # (B, N, q)
        ph = sigmoid(self.hidden_field(s))         # (B, M)
        B = len(s)
        return np.einsum("bia,bm->iam", onehot, ph) / B, onehot.mean(0), ph.mean(0)

    def fit(self, data, epochs=20, lr=0.05, batch=64, k=1, n_chains=64,
            augment=True, rng=None, log_every=0, callback=None):
        """Maximum likelihood (= min KL(data || RBM)) by persistent contrastive divergence.

        augment: randomly relabel the q colours of each sample. This is an exact symmetry of
        any zero-field Ising/Potts Hamiltonian, and repairs datasets where single-flip MC got
        stuck in one ordered sector (otherwise the RBM learns the broken-symmetry bias).
        """
        rng = np.random.default_rng(rng)
        chains = data[rng.choice(len(data), n_chains)].copy()
        for epoch in range(epochs):
            for idx in np.array_split(rng.permutation(len(data)), max(1, len(data) // batch)):
                x = data[idx]
                if augment:
                    perm = np.argsort(rng.random((len(x), self.q)), axis=1)
                    x = perm[np.arange(len(x))[:, None], x]
                for _ in range(k):
                    chains = self.gibbs(chains, rng)
                pos, neg = self._stats(x), self._stats(chains)
                self.W += lr * (pos[0] - neg[0]) * self.mask
                self.b += lr * (pos[1] - neg[1])
                self.c += lr * (pos[2] - neg[2])
            if callback is not None and log_every and (epoch + 1) % log_every == 0:
                callback(epoch + 1, self)
        return self


def local_mask(lat, n_per_site=1, radius=1):
    """Sparse RBM (cf. Pilati & Pieri, PRE 101, 063308 (2020)): hidden units sit on lattice
    sites and connect only to visibles within periodic Manhattan distance <= radius."""
    hidden_site = np.repeat(np.arange(lat.N), n_per_site)
    return lat.distance(np.arange(lat.N)[:, None], hidden_site[None, :]) <= radius


def log_ratio(rbm, lat, beta, s):
    """d(s) = log pi(s) - log p_RBM(s) + const. Std of d over equilibrium samples is the key
    diagnostic: d constant <=> RBM exact <=> every proposal accepted."""
    return -beta * lat.energy(s) + rbm.free_energy(s)


def rbm_mh_step(rbm, lat, beta, s, rng, sites=None, k=1):
    """One RBM-proposed Metropolis-Hastings move. Returns (new_s, accepted, n_changed).

    sites: if given, only these sites take their proposed values (local block move).
    k: Gibbs steps in the proposal. K^k is reversible w.r.t. p_RBM whenever K is, so the
       acceptance formula is unchanged.
    """
    in_block = np.ones(len(s), bool) if sites is None else np.isin(np.arange(len(s)), sites)
    prop = s
    for _ in range(k):
        prop = np.where(in_block, rbm.gibbs(prop, rng), prop)
    if np.log(rng.random()) < log_ratio(rbm, lat, beta, prop) - log_ratio(rbm, lat, beta, s):
        return prop, True, int((prop != s).sum())
    return s, False, 0


def exact_ising_rbm(lat, beta):
    """An RBM that reproduces the Ising Boltzmann weight *exactly* (L. Wang 2017 idea):
    one hidden unit per bond. With u = 2h - 1 = +-1 and t = sign(J_ij),

        sum_u exp(w u (sigma_i + t sigma_j)) = 2 cosh(w (sigma_i + t sigma_j)),

    whose satisfied/unsatisfied ratio cosh(2w)/1 equals exp(2 beta |J|) for
    w = arccosh(exp(2 beta |J|)) / 2. Rewriting u = 2h - 1 gives W = 2w sigma, b = -w sigma.
    Useful as ground truth: d(s) is constant and RBM moves are always accepted.
    """
    assert lat.q == 2
    rbm = RBM(lat.N, 2, len(lat.J), scale=0.0)
    sigma = np.array([-1.0, 1.0])                     # state index -> spin
    w = np.arccosh(np.exp(2 * beta * np.abs(lat.J))) / 2
    bonds = np.arange(len(lat.J))
    for site, sign in ((lat.i, 1.0), (lat.j, np.sign(lat.J))):
        coef = (sign * w)[:, None] * sigma            # (n_bonds, 2)
        rbm.W[site, :, bonds] = 2 * coef
        np.add.at(rbm.b, site, -coef)
    return rbm


# ---------------------------------------------------------------------------------------------
# Bond-gated RBM moves: genuine cluster moves (Fortuin-Kasteleyn gates + RBM-biased colouring).
#
# Add a gate n_b in {0,1} per bond:  p(s, h, n) ∝ exp(-E_RBM(s, h)) prod_b [1 - n_b + n_b (e^K_b - 1) delta_b(s)]
# with delta_b(s) = 1 if bond b is satisfied. Summing out n gives e^{K_b delta_b}, so
#     p_theta(s) ∝ exp(-F(s) + sum_b K_b delta_b(s)),   d(s) = -beta E(s) + F(s) - sum_b K_b delta_b(s).
# Given s, h and n are independent: P(n_b = 1 | s) = delta_b (1 - e^{-K_b}). Given (h, n), every
# open-bond cluster C keeps its internal (satisfied) pattern and picks a colour shift k with weight
# exp(sum_{i in C} l[i, (s_i + k) mod q]). Valid for Ising with any J sign; for q >= 3 only gate J > 0
# bonds (an antiferromagnetic Potts constraint s_i != s_j is not closed under colour shifts).
# Ising trick: K_b = 2 lam beta |J_b| leaves an Ising model at (1 - lam) beta for the RBM, so
# exact_ising_rbm(lat, (1 - lam) beta) is exact for every lam: lam=0 local, lam=1 Swendsen-Wang.
# (Gates: Niedermayer PRL 61, 2026 (1988); Kawashima & Gubernatis cond-mat/9411110;
#  three-body BM -> SW: L. Wang arXiv:1702.08586.)
# ---------------------------------------------------------------------------------------------

def satisfied(lat, s):
    """delta_b(s): 1 where bond b's energy term is at its minimum (J phi > 0)."""
    from model import pair_energy
    return lat.J * pair_energy(s[..., lat.i], s[..., lat.j], lat.q) > 0


def gate_strengths(lat, beta, lam):
    """K_b = lam * (full FK strength): 2 beta |J| for Ising, beta J for ferro Potts bonds, else 0."""
    if lat.q == 2:
        return 2 * lam * beta * np.abs(lat.J)
    return lam * beta * np.clip(lat.J, 0, None)


def clusters(n, i, j):
    """Connected-component labels 0..C-1 of n nodes joined by edges (i, j). Union-find."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(i, j):
        parent[find(a)] = find(b)
    return np.unique([find(a) for a in range(n)], return_inverse=True)[1]


def resample_clusters(rbm, lat, s, h, open_, rng):
    """Sample s' ~ p(s' | h, n): each open-bond cluster picks a colour shift k with
    weight exp(sum_{i in C} l[i, (s_i + k) mod q])."""
    labels = clusters(lat.N, lat.i[open_], lat.j[open_])
    shifted = (s[:, None] + np.arange(lat.q)) % lat.q                         # (N, q)
    site_score = rbm.visible_logits(h)[np.arange(lat.N)[:, None], shifted]    # (N, q)
    score = np.zeros((labels.max() + 1, lat.q))
    np.add.at(score, labels, site_score)
    k = np.argmax(score - np.log(-np.log(rng.random(score.shape))), axis=1)  # Gumbel-max per cluster
    return (s + k[labels]) % lat.q, labels


def gated_log_ratio(rbm, lat, beta, K, s):
    return log_ratio(rbm, lat, beta, s) - (K * satisfied(lat, s)).sum(-1)


def gated_mh_step(rbm, lat, beta, K, s, rng):
    """One bond-gated RBM cluster move with exact MH. Returns (new_s, accepted, n_changed)."""
    h = rbm.sample_h(s, rng)
    open_ = satisfied(lat, s) & (rng.random(len(K)) < 1 - np.exp(-K))
    prop, _ = resample_clusters(rbm, lat, s, h, open_, rng)
    if np.log(rng.random()) < gated_log_ratio(rbm, lat, beta, K, prop) - gated_log_ratio(rbm, lat, beta, K, s):
        return prop, True, int((prop != s).sum())
    return s, False, 0
