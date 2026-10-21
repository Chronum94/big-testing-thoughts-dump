"""Static teacher EOS of bcc-O ice VII/X: 16 H2O (2x2x2 of the 2-molecule Cu2O-type cell).
For each lattice constant a: (1) relax positions at fixed cell starting from ordered ice-rule
(ferroelectric-pattern) asymmetric H, OH=1.0 A, small rattle; (2) single point of symmetric ice X
(H at O-O midpoints). Also a coherent double-well scan E(delta) of H off-centre displacement.
Usage: python teacher_eos.py <ckpt_name> <out.jsonl> [a values...]"""
import os, sys, json, time
import numpy as np
from ase import Atoms, units
from ase.optimize import FIRE, BFGS
from petjax import UPETCalculator

name = sys.argv[1]; out = sys.argv[2]
avals = [float(x) for x in sys.argv[3:]] or list(np.round(np.arange(3.30, 2.29, -0.05), 3))
calc = UPETCalculator.from_checkpoint(f"/home/chronum/scratch/petjax/checkpoints/{name}", stress=True)
NA = 6.02214076e23; MW = 18.01528

def cell2(a, delta):
    """2-molecule cubic cell. O1 (0,0,0), O2 (1/2,1/2,1/2). 4 H at Cu2O sites; delta (A) = shift of
    each H from the O-O midpoint toward one O. H0,H1 -> toward O1 image, H2,H3 -> toward O2: ice rule."""
    O = np.array([[0, 0, 0], [.5, .5, .5]]) * a
    Hsites = np.array([[.25, .25, .25], [.75, .75, .25], [.75, .25, .75], [.25, .75, .75]]) * a
    O1img = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]) * a
    O2 = O[1]
    H = []
    for k in range(4):
        tgt = O1img[k] if k < 2 else O2
        u = tgt - Hsites[k]; u /= np.linalg.norm(u)
        H.append(Hsites[k] + delta * u)
    return Atoms("O2H4", positions=np.vstack([O, H]), cell=[a, a, a], pbc=True)

def press(at):
    s = at.get_stress(voigt=False)  # ASE convention, eV/A^3
    return float(-np.trace(s) / 3 / units.GPa)

def record(d):
    with open(out, "a") as f: f.write(json.dumps(d) + "\n")

for a2 in avals:
    a = a2
    dOO = a * np.sqrt(3) / 2
    nmol = 16
    rho = nmol * MW / NA / ((2 * a) ** 3 * 1e-24)
    # symmetric ice X
    x = cell2(a, 0.0).repeat(2); x.calc = calc
    eX = x.get_potential_energy(); pX = press(x); fX = float(np.abs(x.get_forces()).max())
    # coherent double-well scan
    scan = []
    for dl in np.linspace(0, min(0.5, dOO / 2 - 0.6), 11):
        y = cell2(a, dl).repeat(2); y.calc = calc
        scan.append([float(dl), float(y.get_potential_energy() / nmol)])
    # relax from asymmetric ice VII-like start
    delta0 = max(dOO / 2 - 1.0, 0.0)
    r = cell2(a, delta0).repeat(2); r.rattle(0.02, seed=1); r.calc = calc
    t0 = time.time()
    opt = FIRE(r, logfile=None); conv = opt.run(fmax=0.005, steps=1500)
    pos = r.get_positions(); Oi = [i for i, s in enumerate(r.get_chemical_symbols()) if s == "O"]
    Hi = [i for i, s in enumerate(r.get_chemical_symbols()) if s == "H"]
    dist = r.get_all_distances(mic=True)
    dOH = np.sort(dist[np.ix_(Hi, Oi)], axis=1)  # per H: nearest and 2nd O
    asym = float(np.mean(dOH[:, 1] - dOH[:, 0]))
    rec = dict(model=name, a_conv2=a, a_bcc=a, V_A3_per_H2O=a**3 / 2, V_cm3mol=a**3 / 2 * NA * 1e-24,
               rho_gcc=rho, dOO=dOO,
               E_X_per_mol=eX / nmol, P_X_GPa=pX, fmax_X=fX,
               E_relax_per_mol=r.get_potential_energy() / nmol, P_relax_GPa=press(r),
               stress_relax_GPa=(r.get_stress() / units.GPa).round(3).tolist(),
               relax_converged=bool(conv), relax_steps=opt.nsteps, relax_s=time.time() - t0,
               dOH_near_mean=float(dOH[:, 0].mean()), dOH_far_mean=float(dOH[:, 1].mean()), asym=asym,
               scan=scan)
    record(rec)
    print(f"{name} a={a:.3f} rho={rho:.3f} PX={pX:.1f} Prel={rec['P_relax_GPa']:.1f} "
          f"dE(relax-X)={rec['E_relax_per_mol']-rec['E_X_per_mol']:.4f} asym={asym:.3f} steps={opt.nsteps}", flush=True)
