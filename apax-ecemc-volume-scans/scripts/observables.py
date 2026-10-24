"""Structural, dynamical and stress observables from apax MD trajectories (EC:EMC / LiPF6 boxes), for comparing runs.

usage: [OBS_START=start.extxyz] python observables.py OUT_PREFIX FRAME_FS T_DISCARD_PS RUN_DIR [RUN_DIR ...]
  FRAME_FS: time between saved frames in fs (sampling_rate x dt; apax's h5 "time" is not usable, see nve_drift.py)
Per run prints: T, P; Li+ coordination within 2.6 A by site (carbonyl O, ether O, F) and carbonyl fraction of Li-O
contacts; self-diffusion (COM MSD slope, fit window FIT_PS) for Li+, PF6-, EC, EMC with standard error over molecules.
Writes OUT_PREFIX.png: Li-centred RDFs, normalised VACF and vibrational DOS per element, COM MSDs, shear-stress ACF.
Molecules from the bond graph of the first frame (Li excluded, 1.2 x natural cutoffs); O type by its C neighbours.
"""
import collections
import os
import sys

import h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E401,E702
import numpy as np
from ase import Atoms, units
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

out, frame_fs, t_discard = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); runs = sys.argv[4:]
FIT_PS = tuple(float(x) for x in os.environ.get("OBS_FIT_PS", "1.0,3.5").split(",")); RC = 2.6
import os
start = read(os.environ.get("OBS_START", "/home/chronum/scratch/ecemc/md_radial/start_7722.extxyz"))  # masses come from here (OBS_START for H-mass runs)
sym = np.array(start.get_chemical_symbols()); masses = start.get_masses(); N = len(start)

# molecules and site types from the starting structure
cut = [c * 1.2 if s != "Li" else 0.0 for c, s in zip(natural_cutoffs(start), sym)]
i, j = neighbor_list("ij", start, cut)
_, mol = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(N, N)))
nC = collections.Counter(i[(sym[i] == "O") & (sym[j] == "C")])
o_carb = np.array([k for k in np.where(sym == "O")[0] if nC[k] == 1])
o_eth = np.array([k for k in np.where(sym == "O")[0] if nC[k] == 2])
f_idx, li = np.where(sym == "F")[0], np.where(sym == "Li")[0]
NAMES = {"C3H4O3": "EC", "C4H8O3": "EMC", "C3H6O3": "DMC", "F6P": "PF6", "Li": "Li"}
mol_name = {}
for m_ in np.unique(mol):
    c = collections.Counter(sym[mol == m_]); mol_name[m_] = NAMES.get("".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in sorted(c)), "?")
groups = {n: [m_ for m_, k in mol_name.items() if k == n] for n in ("Li", "PF6", "EC", "EMC")}


def load(d):
    # size by value: znh5md (store="time") leaves step/time one extend() batch short
    f = h5py.File(f"{d}/md.h5", "r"); n = len(f["observables/atoms/potential_energy/value"]); k0 = int(round(t_discard * 1000 / frame_fs))
    g = lambda key: np.asarray(f[key][k0:n])  # noqa: E731
    return g("particles/atoms/position/value"), g("particles/atoms/velocity/value"), g("particles/atoms/box/edges/value"), \
        g("observables/atoms/stress/value") if "observables/atoms/stress/value" in f else None


def acf(x, nmax):
    """mean over the last axes of the time autocorrelation of x (t, ...), lags 0..nmax-1 (FFT, unbiased)."""
    t = x.shape[0]; x = x.reshape(t, -1) - 0.0; F = np.fft.rfft(x, 2 * t, axis=0)
    c = np.fft.irfft(F * np.conj(F), axis=0)[:nmax] / (t - np.arange(nmax))[:, None]
    return c.mean(1)


res = {}
fig, ax = plt.subplots(3, 4, figsize=(19, 12))
for r_i, d in enumerate(runs):
    pos, vel, box, stress = load(d); L = np.diag(box[0]); T_n = len(pos); col = plt.cm.tab10(r_i)
    # apax wrote velocities as momenta / ASE default masses (fixed on branch fix-h5md-masses-time): undo for custom masses
    vel = vel * (Atoms(start.numbers).get_masses() / masses)[None, :, None]
    # unwrap: accumulate minimum-image displacements between consecutive frames
    dpos = np.diff(pos, axis=0); dpos -= L * np.round(dpos / L); unw = np.concatenate([pos[:1], pos[:1] + np.cumsum(dpos, 0)])
    ke = 0.5 * np.einsum("i,fij->f", masses, vel ** 2); T = 2 * ke / (3 * N * units.kB)
    # apax "stress" is dE/d(strain) = stress x volume (eV); pressure = (N k T - tr(dE/de) / 3) / V
    vol_t = np.abs(np.linalg.det(box))
    P = ((2 * ke / 3 - np.trace(stress, axis1=1, axis2=2) / 3) / vol_t / units.GPa) if stress is not None else np.full(T_n, np.nan)
    # Li-centred coordination and RDFs (every 2nd frame)
    cn = collections.defaultdict(list); bins = np.linspace(0, 8, 161); rdf = collections.defaultdict(lambda: np.zeros(160))
    sites = {"O carbonyl": o_carb, "O ether": o_eth, "F": f_idx}
    for k in range(0, T_n, 2):
        dv = pos[k][None, :, :] - pos[k][li][:, None, :]; dv -= L * np.round(dv / L); dist = np.linalg.norm(dv, axis=-1)
        for name, idx in sites.items():
            dd = dist[:, idx]; cn[name].append((dd < RC).sum(1).mean()); rdf[name] += np.histogram(dd, bins)[0]
    vol = np.prod(L); shell = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3); nfr = len(range(0, T_n, 2)); rmid = 0.5 * (bins[1:] + bins[:-1])
    for name, idx in sites.items():
        g = rdf[name] / (nfr * len(li) * len(idx) / vol * shell); ax[0, list(sites).index(name)].plot(rmid, g, color=col, lw=1.2, label=d)
    c_carb, c_eth = np.mean(cn["O carbonyl"]), np.mean(cn["O ether"])
    # COM MSDs and self-diffusion with standard error over molecules
    t_ps = np.arange(T_n) * frame_fs / 1000; lag = np.arange(1, T_n // 2); D = {}
    for gname, mols in groups.items():
        com = np.stack([(unw[:, mol == m_] * masses[mol == m_][None, :, None]).sum(1) / masses[mol == m_].sum() for m_ in mols], 1)
        origins = np.arange(0, T_n - lag[-1], max(1, T_n // 50))
        msd_per = np.stack([((com[origins + l] - com[origins]) ** 2).sum(-1).mean(0) for l in lag])  # (lag, molecule)
        w = (lag * frame_fs / 1000 >= FIT_PS[0]) & (lag * frame_fs / 1000 <= FIT_PS[1])
        slopes = np.polyfit(lag[w] * frame_fs / 1000, msd_per[w], 1)[0] / 6  # A^2/ps per molecule
        D[gname] = (slopes.mean() * 1e-8 / 1e-9, slopes.std(ddof=1) / np.sqrt(len(slopes)) * 1e-8 / 1e-9)  # 1e-9 m^2/s
        ax[2, 0 if gname in ("Li", "PF6") else 1].plot(lag * frame_fs / 1000, msd_per.mean(1), color=col,
                                                         ls="-" if gname in ("Li", "EC") else "--", lw=1.2, label=f"{d} {gname}")
    # VACF / VDOS per element
    nmax = min(T_n // 2, int(1500 / frame_fs))
    for e_i, el in enumerate(("H", "O", "Li")):
        idx = np.where(sym == el)[0]; c = acf(vel[:, idx], nmax); c /= c[0]
        ax[1, e_i].plot(np.arange(nmax) * frame_fs / 1000, c, color=col, lw=1.0, label=d)
        spec = np.abs(np.fft.rfft(c * np.hanning(2 * nmax)[nmax:], 4 * nmax)); freq = np.fft.rfftfreq(4 * nmax, frame_fs * 1e-15) / 2.998e10
        ax[1, 3].plot(freq, spec / spec.max() + e_i, color=col, lw=0.9, label=d if e_i == 0 else None)
    # shear-stress ACF
    if stress is not None:
        sh = np.stack([stress[:, 0, 1], stress[:, 0, 2], stress[:, 1, 2]], 1); c = acf(sh, nmax)
        ax[2, 2].plot(np.arange(nmax) * frame_fs / 1000, c / c[0], color=col, lw=1.0, label=d)
    res[d] = (T.mean(), np.nanmean(P), c_carb, c_eth, np.mean(cn["F"]), c_carb / (c_carb + c_eth), D)
    print(f"{d:22s} T {T.mean():6.1f} K  P {np.nanmean(P):7.3f} GPa | Li CN: O_carb {c_carb:.2f} O_eth {c_eth:.2f} F {np.mean(cn['F']):.2f} "
          f"carb frac {c_carb / (c_carb + c_eth):.3f} | D (1e-9 m2/s): " + " ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in D.items()), flush=True)
for a, t in zip(ax[0, :3], ("g(r) Li – O carbonyl", "g(r) Li – O ether", "g(r) Li – F")):
    a.set(title=t, xlabel="r (Å)", xlim=(1.5, 7)); a.legend(fontsize=7)
for a, el in zip(ax[1, :3], ("H", "O", "Li")):
    a.set(title=f"normalised VACF, {el}", xlabel="t (ps)"); a.axhline(0, color="k", lw=0.5)
ax[1, 3].set(title=f"VDOS H / O / Li (offset); Nyquist {0.5 / (frame_fs * 1e-15) / 2.998e10:.0f} cm⁻¹", xlabel="cm⁻¹")
ax[2, 0].set(title="COM MSD: Li+ (solid), PF6- (dashed)", xlabel="lag (ps)", ylabel="Å²"); ax[2, 1].set(title="COM MSD: EC (solid), EMC (dashed)", xlabel="lag (ps)")
ax[2, 2].set(title="shear-stress ACF (normalised)", xlabel="t (ps)"); ax[2, 2].axhline(0, color="k", lw=0.5)
for a in (ax[2, 0], ax[2, 1], ax[2, 2], ax[1, 3]):
    a.legend(fontsize=6)
ax[0, 3].axis("off"); ax[2, 3].axis("off")
fig.suptitle(f"{out}: observables (discard first {t_discard} ps, frames every {frame_fs} fs)"); fig.tight_layout(); fig.savefig(f"{out}.png", dpi=100); print("wrote", f"{out}.png")
