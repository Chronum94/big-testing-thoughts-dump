"""Classical Ising / q-state Potts models on periodic d-dimensional hypercubic lattices.

Spins are integers s_i in {0, ..., q-1}. Every bond <ij> carries its own coupling J_ij,
so disorder and staggered patterns are just different J arrays.

    q == 2 (Ising):  E(s) = -sum_<ij> J_ij sigma_i sigma_j,  sigma = 2s - 1
    q >= 3 (Potts):  E(s) = -sum_<ij> J_ij delta(s_i, s_j)

J > 0 is ferromagnetic, J < 0 antiferromagnetic. 2D Ising, uniform J=1: T_c = 2.269.
"""
import numpy as np


def pair_energy(a, b, q):
    """phi(a, b) such that E = -sum J phi. Works elementwise on arrays."""
    same = (a == b)
    return 2.0 * same - 1.0 if q == 2 else same.astype(float)


def make_bonds(L, d, kind="uniform", J=1.0, p=0.5, J2=0.5, rng=None):
    """Nearest-neighbour bonds of a periodic L^d lattice.

    Returns (i, j, Jij, coords): bond endpoints, couplings, and site coordinates (N, d).

    kind:
      uniform  every bond J
      pm       +J with probability p, else -J  (p=0.5: Edwards-Anderson +-J spin glass)
      gauss    J_ij ~ Normal(0, J)               (Gaussian spin glass)
      dilute   J with probability p, else 0      (bond dilution)
      ffim     Villain fully frustrated Ising model (2D, even L)
      staggered  bond from an even coordinate along its axis gets J, from an odd one J2
                 (dimerized / two-sublattice bonding pattern; needs even L)
    """
    rng = np.random.default_rng(rng)
    site = np.arange(L**d).reshape([L] * d)
    coords = np.stack(np.unravel_index(site.ravel(), site.shape), axis=1)
    i, j, axis = [], [], []
    for ax in range(d):
        i.append(site.ravel())
        j.append(np.roll(site, -1, axis=ax).ravel())
        axis.append(np.full(site.size, ax))
    i, j, axis = map(np.concatenate, (i, j, axis))

    nb = len(i)
    if kind == "uniform":
        Jij = np.full(nb, J)
    elif kind == "pm":
        Jij = np.where(rng.random(nb) < p, J, -J)
    elif kind == "gauss":
        Jij = rng.normal(0.0, J, nb)
    elif kind == "dilute":
        Jij = np.where(rng.random(nb) < p, J, 0.0)
    elif kind == "ffim":
        # Villain fully frustrated model: axis-0 bonds ferro, axis-1 bonds alternate sign with the
        # axis-0 coordinate, so every plaquette has exactly one AF bond (needs even L, d=2)
        assert d == 2 and L % 2 == 0
        Jij = np.where((axis == 1) & (coords[i, 0] % 2 == 1), -J, J)
    elif kind == "staggered":
        assert L % 2 == 0, "staggered bonds need even L to be consistent with PBC"
        Jij = np.where(coords[i, axis] % 2 == 0, J, J2)
    else:
        raise ValueError(f"unknown bond kind {kind!r}")
    return i, j, Jij.astype(float), coords


class Lattice:
    """Holds the bond list plus a per-site neighbour table for fast local energy changes."""

    def __init__(self, L, d, q=2, **bond_kwargs):
        self.L, self.d, self.q, self.N = L, d, q, L**d
        self.i, self.j, self.J, self.coords = make_bonds(L, d, **bond_kwargs)
        # Neighbour table: each site appears in exactly 2d bonds on a periodic hypercube.
        # ponytail: assumes L >= 3 so the 2d neighbours are distinct; L=2 double-counts bonds
        # consistently in both energy() and delta_energy(), so it stays correct, just odd.
        nbrs = [[] for _ in range(self.N)]
        for a, b, Jab in zip(self.i, self.j, self.J):
            nbrs[a].append((b, Jab))
            nbrs[b].append((a, Jab))
        self.nbr = np.array([[n for n, _ in row] for row in nbrs])
        self.nbrJ = np.array([[J for _, J in row] for row in nbrs])

    def energy(self, s):
        """Total energy; s has shape (N,) or (batch, N)."""
        s = np.asarray(s)
        return -(self.J * pair_energy(s[..., self.i], s[..., self.j], self.q)).sum(-1)

    def delta_energy(self, s, site, new):
        """E(s with s[site]=new) - E(s)."""
        nb, Jn = self.nbr[site], self.nbrJ[site]
        return -(Jn * (pair_energy(new, s[nb], self.q) - pair_energy(s[site], s[nb], self.q))).sum()

    def distance(self, a, b):
        """Periodic Manhattan distance between sites a and b (broadcasts)."""
        dx = np.abs(self.coords[a] - self.coords[b])
        return np.minimum(dx, self.L - dx).sum(-1)

    def order_parameter(self, s):
        """Potts magnetisation m = (q * max_a n_a / N - 1) / (q - 1), in [0, 1]; |m| for Ising.

        Meaningful for ferromagnets only; for glasses/antiferromagnets use E or an overlap.
        """
        s = np.atleast_2d(s)
        counts = np.stack([(s == a).sum(-1) for a in range(self.q)], -1)
        return (self.q * counts.max(-1) / self.N - 1) / (self.q - 1)


def metropolis_sweep(lat, s, beta, rng):
    """One sweep = N random-site single-spin Metropolis attempts. Modifies s in place."""
    sites = rng.integers(lat.N, size=lat.N)
    # Propose a different state uniformly: shift by 1..q-1 (symmetric proposal).
    shifts = rng.integers(1, lat.q, size=lat.N)
    u = rng.random(lat.N)
    acc = 0
    for site, shift, r in zip(sites, shifts, u):
        new = (s[site] + shift) % lat.q
        dE = lat.delta_energy(s, site, new)
        if dE <= 0 or r < np.exp(-beta * dE):
            s[site] = new
            acc += 1
    return acc / lat.N


def generate_dataset(lat, beta, n_samples, n_therm=1000, thin=10, rng=None, s=None):
    """Single-flip Metropolis samples (n_samples, N) taken every `thin` sweeps.

    The samples are correlated if thin < tau_int; check it with tau_int() in run.py.
    """
    rng = np.random.default_rng(rng)
    s = rng.integers(lat.q, size=lat.N) if s is None else s.copy()
    for _ in range(n_therm):
        metropolis_sweep(lat, s, beta, rng)
    data = np.empty((n_samples, lat.N), dtype=np.int64)
    for k in range(n_samples):
        for _ in range(thin):
            metropolis_sweep(lat, s, beta, rng)
        data[k] = s
    return data


def wolff_step(lat, s, beta, rng):
    """One Wolff single-cluster update for Ising (q=2) with arbitrary-sign bonds: grow along
    *satisfied* bonds with p = 1 - exp(-2 beta |J|), flip the cluster. In place; returns size.
    Ground-truth cluster move to compare RBM proposals against (inefficient for spin glasses,
    whose FK clusters percolate, but still exact)."""
    assert lat.q == 2
    seed = rng.integers(lat.N)
    in_cluster = np.zeros(lat.N, bool)
    in_cluster[seed] = True
    stack = [seed]
    while stack:
        i = stack.pop()
        for j, Jij in zip(lat.nbr[i], lat.nbrJ[i]):
            satisfied = Jij * (2 * (s[i] == s[j]) - 1) > 0
            if not in_cluster[j] and satisfied and rng.random() < 1 - np.exp(-2 * beta * abs(Jij)):
                in_cluster[j] = True
                stack.append(j)
    s[in_cluster] ^= 1
    return int(in_cluster.sum())
