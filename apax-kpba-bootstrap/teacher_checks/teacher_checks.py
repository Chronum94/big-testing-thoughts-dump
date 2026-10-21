"""Cheap PET-MAD teacher checks for cubic K_x Fe[Fe(CN)6].

Usage: python teacher_checks.py <model: s|m> <outjson>
Builds 64-atom (K2) and 60-atom (K1) Fm-3m conventional cells, relaxes positions+cell
(FrechetCellFilter + FIRE, fmax 0.02), reports a, Fe-C, C-N, Fe-N, K positions,
then C-N finite-difference stretch (+-0.05 A) and Gamma-point vibrational
frequencies of the relaxed cell (CN stretch band).
"""
import os, sys, json, time
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.units import _hplanck, _c, _amu, _e
from petjax import UPETCalculator

model = sys.argv[1]; out = sys.argv[2]
ck = f"/home/chronum/scratch/petjax/checkpoints/pet-mad-{model}"
calc = UPETCalculator.from_checkpoint(ck, stress=True, default_dtype="float64")

FCC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def build(nK, a=10.2, xC=0.188, xN=0.301):
    sym, pos = [], []
    for t in FCC:  # Fe(C) 4a, Fe(N) 4b
        sym += ["Fe"]; pos.append(t)
        sym += ["Fe"]; pos.append(t + [.5, .5, .5])
    for x, el in ((xC, "C"), (xN, "N")):  # 24e (x,0,0)
        for t in FCC:
            for v in ([x, 0, 0], [-x, 0, 0], [0, x, 0], [0, -x, 0], [0, 0, x], [0, 0, -x]):
                sym.append(el); pos.append(t + v)
    # 8c: (1/4,1/4,1/4)+fcc and (3/4,3/4,3/4)+fcc
    k8 = [t + [.25, .25, .25] for t in FCC] + [t + [.75, .75, .75] for t in FCC]
    if nK == 8:
        ks = k8
    elif nK == 4:
        ks = k8[:4]  # one tetrahedral subset -> F-43m ordering (KFe[Fe(CN)6])
    elif nK == 0:
        ks = []
    for k in ks:
        sym.append("K"); pos.append(k)
    at = Atoms(sym, scaled_positions=np.mod(pos, 1.0), cell=np.eye(3) * a, pbc=True)
    return at


def bonds(at):
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", at, 2.4)
    s = at.get_chemical_symbols()
    # distinguish Fe(C) from Fe(N) by neighbour type
    res = {"Fe-C": [], "C-N": [], "Fe-N": []}
    for a_, b_, dd in zip(i, j, d):
        p = s[a_] + "-" + s[b_]
        if p in res:
            res[p].append(dd)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in res.items() if v}


def cn_stretch(at, h=0.05):
    """Stretch one C-N pair symmetrically along the bond (centre of mass of pair fixed)."""
    s = at.get_chemical_symbols()
    iC = s.index("C")
    from ase.neighborlist import neighbor_list
    i, j, D = neighbor_list("ijD", at, 1.4)
    jN = [jj for ii, jj in zip(i, j) if ii == iC and s[jj] == "N"][0]
    vec = at.get_distance(iC, jN, mic=True, vector=True); u = vec / np.linalg.norm(vec)
    mC, mN = 12.011, 14.007; mu = mC * mN / (mC + mN)
    E = []
    ds = np.linspace(-h, h, 11)
    for dd in ds:
        b = at.copy(); b.calc = calc
        b.positions[iC] -= u * dd * mN / (mC + mN)
        b.positions[jN] += u * dd * mC / (mC + mN)
        E.append(b.get_potential_energy())
    E = np.array(E)
    c = np.polyfit(ds, E - E[5], 4)
    k = 2 * c[2]  # eV/A^2
    kSI = k * _e / 1e-20
    nu = np.sqrt(kSI / (mu * _amu)) / (2 * np.pi * _c * 100)
    # also mass-free local fit with pure harmonic over +-0.02
    c2 = np.polyfit(ds[3:8], E[3:8] - E[5], 2)
    nu2 = np.sqrt(2 * c2[0] * _e / 1e-20 / (mu * _amu)) / (2 * np.pi * _c * 100)
    return dict(k_eV_A2=float(k), nu_cm1_quartic_fit=float(nu), nu_cm1_harm_pm0p02=float(nu2),
                cubic_coeff=float(c[1]), d=ds.tolist(), dE=(E - E[5]).tolist())


def gamma_freqs(at, delta=0.01):
    from ase.vibrations import Vibrations
    b = at.copy(); b.calc = calc
    import tempfile
    d = tempfile.mkdtemp(dir="/home/chronum/scratch/pba/lit/vibtmp") if os.path.isdir("/home/chronum/scratch/pba/lit/vibtmp") else None
    os.makedirs("/home/chronum/scratch/pba/lit/vibtmp", exist_ok=True)
    name = f"/home/chronum/scratch/pba/lit/vibtmp/{model}_{len(b)}"
    v = Vibrations(b, delta=delta, nfree=2, name=name)
    v.run()
    f = v.get_frequencies().real
    v.clean()
    return np.sort(f)


results = {"model": model}
for label, nK in (("K2Fe[Fe(CN)6]", 8), ("KFe[Fe(CN)6]", 4), ("Fe[Fe(CN)6]", 0)):
    t0 = time.time()
    at = build(nK); at.calc = calc
    opt = FIRE(FrechetCellFilter(at), logfile=None)
    conv = opt.run(fmax=0.02, steps=1500)
    cell = at.cell.cellpar()
    r = dict(n=len(at), converged=bool(conv), steps=opt.nsteps, cellpar=cell.tolist(),
             a_mean=float(np.mean(cell[:3])), vol_per_fu=float(at.get_volume() / 4),
             E_per_atom=float(at.get_potential_energy() / len(at)),
             stress_GPa=(at.get_stress() / 0.006241509).tolist(),
             bonds=bonds(at), time_relax_s=time.time() - t0)
    if nK:
        sp = at.get_scaled_positions()[np.array(at.get_chemical_symbols()) == "K"]
        r["K_scaled_first"] = sp[:2].tolist()
        # displacement of K from ideal 8c
        ideal = build(nK).get_scaled_positions()[np.array(at.get_chemical_symbols()) == "K"]
        dfrac = (sp - ideal + 0.5) % 1 - 0.5
        r["K_offcentre_A"] = float(np.max(np.linalg.norm(dfrac @ at.cell.array, axis=1)))
    at.write(f"/home/chronum/scratch/pba/lit/relaxed_{model}_{label.replace('[','').replace(']','').replace('(','').replace(')','')}.extxyz")
    r["cn_stretch"] = cn_stretch(at)
    print(label, json.dumps(r)[:600], flush=True)
    if nK == 8 or (model == "s"):
        t1 = time.time()
        f = gamma_freqs(at)
        r["gamma_top12_cm1"] = f[-12:].tolist()
        hi = f[f > 1800]
        r["gamma_CN_band_cm1"] = [float(hi.min()), float(hi.max()), int(len(hi))] if len(hi) else None
        r["gamma_lowest6_cm1"] = f[:9].tolist()
        r["time_vib_s"] = time.time() - t1
    results[label] = r
    json.dump(results, open(out, "w"), indent=1)
print("DONE")
