"""Short teacher NVT MD of bcc-O ice (54 H2O = 3x3x3 of 2-molecule cell) to probe superionic behaviour.
Usage: python teacher_md.py <ckpt> <a2_A> <T_K> <nsteps> <dt_fs> <out_prefix>
Writes <out>.jsonl every 20 steps: t_fs, T, P_GPa(virial, no kinetic), E_pot, MSD_O, MSD_H (A^2, unwrapped, COM-corrected)."""
import sys, json, time
import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import write
from petjax import UPETCalculator

ck, a, T, nsteps, dt, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]

def cell2(a, delta):
    O = np.array([[0, 0, 0], [.5, .5, .5]]) * a
    Hs = np.array([[.25, .25, .25], [.75, .75, .25], [.75, .25, .75], [.25, .75, .75]]) * a
    O1 = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]) * a
    H = []
    for k in range(4):
        t = O1[k] if k < 2 else O[1]
        u = t - Hs[k]; u /= np.linalg.norm(u); H.append(Hs[k] + delta * u)
    return Atoms("O2H4", positions=np.vstack([O, H]), cell=[a] * 3, pbc=True)

at = cell2(a, max(a * np.sqrt(3) / 4 - 1.0, 0)).repeat(3)
at.calc = UPETCalculator.from_checkpoint(f"/home/chronum/scratch/petjax/checkpoints/{ck}", stress=True)
MaxwellBoltzmannDistribution(at, temperature_K=T, rng=np.random.default_rng(0)); Stationary(at)
dyn = Langevin(at, dt * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=np.random.default_rng(1))
isO = np.array(at.get_chemical_symbols()) == "O"; m = at.get_masses()
ref = at.get_positions().copy(); unwr = ref.copy(); prev = at.get_scaled_positions().copy()
f = open(out + ".jsonl", "w"); t0 = time.time()
def log():
    global prev, unwr
    s = at.get_scaled_positions(); ds = s - prev; ds -= np.round(ds); prev = s
    unwr = unwr + ds @ at.cell.array
    d = unwr - ref; d -= (m[:, None] * d).sum(0) / m.sum()
    msd = (d ** 2).sum(1)
    st = at.get_stress(voigt=False, include_ideal_gas=False)
    rec = dict(step=dyn.nsteps, t_fs=dyn.nsteps * dt, T=at.get_temperature(),
               P_vir_GPa=float(-np.trace(st) / 3 / units.GPa), Epot=at.get_potential_energy(),
               msd_O=float(msd[isO].mean()), msd_H=float(msd[~isO].mean()), wall=time.time() - t0)
    f.write(json.dumps(rec) + "\n"); f.flush()
dyn.attach(log, interval=20)
dyn.attach(lambda: write(out + ".extxyz", at, append=True), interval=200)
dyn.run(nsteps)
