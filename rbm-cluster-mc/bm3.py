"""Wang's semi-restricted Boltzmann machines, whose hidden units couple to *products* of spins.
L. Wang, PRE 96, 051301(R) (2017), arXiv:1702.08586. General form (Eq. 10):

    E(s, h) = E0(s) - sum_a [W_a F_a(s) + b_a] h_a,    h_a in {0, 1}

Contrast with rbm.py, where hidden units couple to a *linear* function of the spins: there the spins
are independent given h, so moves are local (explore/NOTES.md §1, §7). Here F_a is a product (s_i s_j) or a
sum of products. Given h, the spins form clusters, or an effective Ising model that one samples with
clusters. Ising only (q = 2); spins s in {0,1}, sigma = 2s - 1.
"""
import numpy as np

from rbm import clusters, sigmoid


def spin(s):
    return 2 * s - 1


def bond_products(lat, s):
    sg = spin(s)
    return sg[..., lat.i] * sg[..., lat.j]


# --- Link BM (Eqs. 3-5): the Kandel-Domany / Niedermayer family; b -> -inf is Swendsen-Wang -----

def solve_ratio(r, b):
    """log x for x > 0 solving (1 + e^b x) / (1 + e^b / x) = r, i.e. e^b x^2 + (1 - r) x - r e^b = 0."""
    eb = np.exp(b)
    disc = np.sqrt((1 - r) ** 2 + 4 * r * eb**2)
    # both forms are the positive root; pick the one without cancellation
    x = np.where(r >= 1, ((r - 1) + disc) / (2 * eb), 2 * r * eb / np.maximum((1 - r) + disc, 1e-300))
    return np.log(x)


def link_bm_weights(lat, beta, b):
    """W_l solving (1 + e^{b+W}) / (1 + e^{b-W}) = e^{2 beta J_l} (Eq. 5), so that the BM marginal
    p(s) = prod_l (1 + e^{W_l s_i s_j + b}) equals the Ising weight. Any sign of J_l; W_l has J's sign."""
    return solve_ratio(np.exp(2 * beta * lat.J), b)


def link_bm_logp(lat, W, b, s):
    return np.logaddexp(0.0, W * bond_products(lat, s) + b).sum(-1)


def link_bm_step(lat, beta, W, b, s, rng):
    """h_l ~ sigmoid(W_l s_i s_j + b) (Eq. 4); flip each component joined by active links with prob 1/2;
    MH-accept with Eq. (1). Rejection-free when W solves Eq. (5). Returns (new_s, accepted, n_changed)."""
    h = rng.random(len(lat.J)) < sigmoid(W * bond_products(lat, s) + b)
    labels = clusters(lat.N, lat.i[h], lat.j[h])
    prop = s ^ (rng.random(labels.max() + 1) < 0.5)[labels]
    d = lambda x: -beta * lat.energy(x) - link_bm_logp(lat, W, b, x)
    if np.log(rng.random()) < d(prop) - d(s):
        return prop, True, int((prop != s).sum())
    return s, False, 0


# --- Plaquette Ising model and its BM (Eqs. 6-9), 2D square lattice -----------------------------

def plaquettes(lat):
    """Corner sites (P, 4) as a, a+x, a+x+y, a+y, and the two parallel-link pairs (P, 2, 2) as bond
    indices. Bond index = axis * N + site (make_bonds layout)."""
    assert lat.d == 2
    N, L = lat.N, lat.L
    site = np.arange(N).reshape(L, L)
    nxt = lambda p, ax: np.roll(site, -1, axis=ax).ravel()[p]
    a = np.arange(N)
    b, d = nxt(a, 0), nxt(a, 1)
    c = nxt(b, 1)
    corners = np.stack([a, b, c, d], 1)
    pairs = np.stack([np.stack([a, d], 1),            # axis-0 links (a,b) and (d,c)
                      np.stack([N + a, N + b], 1)],   # axis-1 links (a,d) and (b,c)
                     1)
    return corners, pairs


def plaquette_energy(lat, s, K):
    """E = -sum_l J_l s_i s_j - K sum_p s_a s_b s_c s_d   (Eq. 6 times -1/beta)."""
    corners, _ = plaquettes(lat)
    return lat.energy(s) - K * spin(s)[..., corners].prod(-1).sum(-1)


def plaquette_bm_step(lat, beta, K, s, rng):
    """Wang's rejection-free cluster update for the plaquette Ising model (K > 0):
    1. per plaquette pick a random parallel-link pair (l, l'), F = s_l + s_l' (bond products);
    2. h_p ~ sigmoid(W F) with W = arccosh(e^{2 beta K})   (Eqs. 7, 9);
    3. given h this is an Ising model with couplings beta J_l + sum_p W (h_p - 1/2) over the
       plaquettes that picked l (Eq. 8). One Swendsen-Wang step on it (valid for any sign).
    Note: a link shared by two plaquettes that both picked it with h=0 gets beta J - W, which is
    antiferromagnetic once W > beta J, so the conditional model can be frustrated (see explore/NOTES.md)."""
    assert K > 0
    _, pairs = plaquettes(lat)
    P = len(pairs)
    pick = pairs[np.arange(P), rng.integers(2, size=P)]            # (P, 2) bond indices
    prods = bond_products(lat, s)
    W = np.arccosh(np.exp(2 * beta * K))
    h = rng.random(P) < sigmoid(W * prods[pick].sum(1))
    Keff = beta * lat.J.copy()
    np.add.at(Keff, pick.ravel(), np.repeat(W * (h - 0.5), 2))
    active = (Keff * prods > 0) & (rng.random(len(Keff)) < 1 - np.exp(-2 * np.abs(Keff)))
    labels = clusters(lat.N, lat.i[active], lat.j[active])
    return s ^ (rng.random(labels.max() + 1) < 0.5)[labels], Keff


def plaquette_metropolis_sweep(lat, s, beta, K, rng):
    """Local baseline for Eq. (6): N random-site single-flip Metropolis attempts. In place."""
    corners, _ = plaquettes(lat)
    site_plaq = np.argsort(corners.ravel(), kind="stable").reshape(lat.N, 4) // 4  # the 4 plaquettes of each site
    for i, r in zip(rng.integers(lat.N, size=lat.N), rng.random(lat.N)):
        sg = spin(s)
        local = (lat.nbrJ[i] * sg[lat.nbr[i]]).sum() + K * (sg[corners[site_plaq[i]]].prod(1) * sg[i]).sum()
        dE = 2 * sg[i] * local      # flipping s_i changes every term containing it by -2x
        if dE <= 0 or r < np.exp(-beta * dE):
            s[i] ^= 1


# --- Frustrated plaquette BM (Wang App. B, generalised to +-J): KBD clusters ------------------------
# Checkerboard plaquettes (x + y even) share only corners, so every bond lies in exactly one of them and
# pi = exp(beta |J| sum_p F_p) with F_p = sum_{b in p} sign(J_b) s_i s_j = 2 (#satisfied) - 4.
# * Frustrated plaquette (odd # of AF bonds, F = +-2): hidden h_p ~ sigmoid(W F_p + b). If h_p = 1, freeze
#   the parallel pair whose two bonds share the majority status (both satisfied if F = 2, both unsatisfied
#   if F = -2). Flipping one rigid dimer against the other toggles both cross bonds, so F_p is conserved
#   (Eq. A2). Exact when (1 + e^{b+2W}) / (1 + e^{b-2W}) = e^{4 beta |J|}; b -> -inf is
#   Kandel-Ben-Av-Domany (PRL 65, 941 (1990)), the reference cluster algorithm for the FFIM.
# * Unfrustrated plaquette: one hidden unit has no exact (W, b) for 4 beta |J| > ln 2.618, so its bonds
#   get Swendsen-Wang link gates instead (freeze each satisfied bond w.p. 1 - e^{-2 beta |J|}).
# Then flip each frozen component with probability 1/2, and MH-accept for safety (always accepts if W is exact).

def checkerboard(lat):
    """Corner-sharing plaquettes: their 4 bond indices (P/2, 4) as [pair0 a, pair0 b, pair1 a, pair1 b]."""
    corners, pairs = plaquettes(lat)
    x, y = lat.coords[corners[:, 0]].T
    keep = (x + y) % 2 == 0
    assert lat.L % 2 == 0, "checkerboard decomposition needs even L"
    return pairs[keep].reshape(-1, 4)


def kbd_logp(lat, beta, W, b, s, cb=None):
    """log of the BM marginal (up to a constant): frustrated plaquettes log(1 + e^{W F + b}),
    unfrustrated ones beta |J| F (their link gates are exact FK)."""
    cb = checkerboard(lat) if cb is None else cb
    sgn = np.sign(lat.J)[cb]
    frus = sgn.prod(1) < 0
    F = (sgn * bond_products(lat, s)[..., cb]).sum(-1)
    absJ = np.abs(lat.J[0])
    return (np.logaddexp(0, W * F + b) * frus).sum(-1) + (beta * absJ * F * ~frus).sum(-1)


def kbd_weights(lat, beta, b):
    """W for frustrated plaquettes making the BM exact:  e^{2W} solves the Eq. 5 form with r = e^{4 beta |J|}."""
    return 0.5 * solve_ratio(np.exp(4 * beta * np.abs(lat.J[0])), b)


def kbd_proposal(lat, beta, W, b, s, rng, cb=None):
    """Returns (proposal, frozen-bond mask). See the block comment above."""
    assert np.allclose(np.abs(lat.J), np.abs(lat.J[0])), "needs uniform |J| (+-J or FFIM)"
    cb = checkerboard(lat) if cb is None else cb
    sgn = np.sign(lat.J)[cb]
    frus = sgn.prod(1) < 0
    sat = lat.J * bond_products(lat, s) > 0                     # (n_bonds,)
    sat_p = sat[cb]                                             # (P, 4)
    F = 2 * sat_p.sum(1) - 4
    frozen = np.zeros(len(lat.J), bool)
    # frustrated plaquettes: KBD freezing of the majority-status parallel pair
    h = frus & (rng.random(len(cb)) < sigmoid(W * F + b))
    majority = np.where((F > 0)[:, None], sat_p, ~sat_p)        # bonds with the majority status
    pair0 = majority[:, 0] & majority[:, 1]                     # parallel pair 0 both majority
    pick = np.where(pair0[:, None], cb[:, :2], cb[:, 2:])       # exactly one pair qualifies when |F| = 2
    frozen[pick[h].ravel()] = True
    # unfrustrated plaquettes: SW gates on satisfied bonds
    sw = cb[~frus].ravel()
    absJ = np.abs(lat.J[0])
    frozen[sw] = sat[sw] & (rng.random(len(sw)) < 1 - np.exp(-2 * beta * absJ))
    labels = clusters(lat.N, lat.i[frozen], lat.j[frozen])
    return s ^ (rng.random(labels.max() + 1) < 0.5)[labels], frozen


def kbd_step(lat, beta, W, b, s, rng, cb=None):
    """One KBD-BM cluster move with MH. Returns (new_s, accepted, n_changed)."""
    cb = checkerboard(lat) if cb is None else cb
    prop, _ = kbd_proposal(lat, beta, W, b, s, rng, cb)
    d = lambda x: -beta * lat.energy(x) - kbd_logp(lat, beta, W, b, x, cb)
    if np.log(rng.random()) < d(prop) - d(s):
        return prop, True, int((prop != s).sum())
    return s, False, 0


# --- Feature-learning BM (Wang Eq. 10 with a dictionary of candidate features) ----------------------
# Candidate features, each with its own hidden unit and (W, b):
#   link (per bond):        F = sign(J) s_i s_j                -> when active, freeze the bond;
#   pair0 / pair1 (per plaquette): sum of two parallel signed bonds -> freeze both bonds;
#   plaq (per plaquette, both checkerboard sublattices): sum of the four signed bonds ->
#        freeze the majority parallel pair if frustrated (KBD rule), else freeze the whole plaquette.
# Each freezing rule conserves its own feature, so cluster flips keep E(s,h) = -sum (W F + b) h unchanged and the
# move is reversible w.r.t. p_BM(s) = prod_features (1 + e^{W F + b})  (Eqs. A2/A3). MH with
# d = -beta E - log p_BM makes it exact for ANY parameters.
# Params: dict W, b of shape (P, 3) [pair0, pair1, plaq] and Wl, bl of shape (n_bonds,).
# Two parametrisations: `typed` (12 shared types, free) and `exact_split` (rejection-free by construction).

OFF = -30.0


class FeatureBM:
    N_TYPES = 12

    def __init__(self, lat):
        corners, pairs = plaquettes(lat)
        self.bonds = pairs.reshape(-1, 4)                           # [pair0 a, pair0 b, pair1 a, pair1 b]
        x, y = lat.coords[corners[:, 0]].T
        self.sub = (x + y) % 2
        self.frus = np.sign(lat.J)[self.bonds].prod(1) < 0
        self.type = np.stack([4 * k + 2 * self.sub + self.frus for k in range(3)], 1)   # (P, 3)
        self.absJ = np.abs(lat.J[0])
        assert np.allclose(np.abs(lat.J), self.absJ), "needs uniform |J| (+-J or FFIM)"

    def features(self, lat, s):
        sl = np.sign(lat.J) * bond_products(lat, s)                              # (..., n_bonds), +1 = satisfied
        sb = sl[..., self.bonds]                                                  # (..., P, 4)
        return np.stack([sb[..., 0] + sb[..., 1], sb[..., 2] + sb[..., 3], sb.sum(-1)], -1), sb, sl

    def logp(self, lat, prm, s):
        F, _, sl = self.features(lat, s)
        return (np.logaddexp(0, prm["W"] * F + prm["b"]).sum((-1, -2))
                + np.logaddexp(0, prm["Wl"] * sl + prm["bl"]).sum(-1))

    def proposal(self, lat, prm, s, rng):
        F, sb, sl = self.features(lat, s)
        h = rng.random(F.shape) < sigmoid(prm["W"] * F + prm["b"])               # (P, 3)
        frozen = rng.random(len(lat.J)) < sigmoid(prm["Wl"] * sl + prm["bl"])    # link features
        frozen[self.bonds[h[:, 0], :2].ravel()] = True
        frozen[self.bonds[h[:, 1], 2:].ravel()] = True
        majority = np.where((F[:, 2] > 0)[:, None], sb > 0, sb < 0)
        pick = np.where((majority[:, 0] & majority[:, 1])[:, None], self.bonds[:, :2], self.bonds[:, 2:])
        frozen[pick[h[:, 2] & self.frus].ravel()] = True
        frozen[self.bonds[h[:, 2] & ~self.frus, :3].ravel()] = True              # a-b, d-c, a-d: rigid
        labels = clusters(lat.N, lat.i[frozen], lat.j[frozen])
        return s ^ (rng.random(labels.max() + 1) < 0.5)[labels], frozen

    def step(self, lat, beta, prm, s, rng):
        """One feature-BM cluster move with MH. Returns (new_s, accepted, n_changed)."""
        prop, _ = self.proposal(lat, prm, s, rng)
        d = lambda x: -beta * lat.energy(x) - self.logp(lat, prm, x)
        if np.log(rng.random()) < d(prop) - d(s):
            return prop, True, int((prop != s).sum())
        return s, False, 0

    def typed(self, theta):
        """Free parametrisation: theta = [W_0..W_11, b_0..b_11] shared per kind x sublattice x frustrated; links off."""
        nb = len(self.frus) * 2
        return dict(W=theta[:12][self.type], b=theta[12:][self.type], Wl=np.zeros(nb), bl=np.full(nb, OFF))

    def exact_split(self, lat, beta, a0, a1, b0, b1, bl):
        """Rejection-free family. Each bond's coupling beta|J| is split: share a_s to the sublattice-s plaquette
        containing it (only if that plaquette is frustrated; frustrated plaquette features are exact for any
        share, see kbd_weights), the rest to its own link feature (exact via Eq. 5). Pair features are off:
        they cannot be exact at low T (two conditions, like an unfrustrated plaquette). Needs a0, a1 >= 0, a0+a1 <= 1."""
        P = len(self.frus)
        share = np.where(self.sub == 0, a0, a1) * self.frus                       # (P,) plaquette shares
        W = np.zeros((P, 3)); b = np.full((P, 3), OFF)
        r = np.exp(4 * share * beta * self.absJ)
        b[:, 2] = np.where(self.sub == 0, b0, b1)
        W[:, 2] = 0.5 * solve_ratio(r, b[:, 2])
        b[share == 0, 2] = OFF
        link_share = np.ones(len(lat.J))
        np.subtract.at(link_share, self.bonds.ravel(), np.repeat(share, 4))       # each bond: 1 - its plaquette shares
        Wl = solve_ratio(np.exp(2 * beta * self.absJ * link_share), bl)           # feature is already sign(J) s s
        return dict(W=W, b=b, Wl=Wl, bl=np.full(len(lat.J), float(bl)))

    def kbd(self, lat, beta):
        """KBD (Wang App. B, b -> -inf) on checkerboard sublattice 0, as a point of exact_split."""
        return self.exact_split(lat, beta, 1.0, 0.0, OFF, OFF, OFF)
