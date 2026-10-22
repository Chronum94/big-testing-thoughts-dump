"""Teacher (PET-MAD-m) Langevin MD of a packed electrolyte box; writes labelled frames as it goes.
usage: python teacher_md.py START.extxyz OUT.extxyz T_K PS [SEED]   (SEED shifts the velocity and Langevin RNGs; default 0)
FIRE pre-relax (100 steps, fmax 0.5); fast sampling: H mass 4 amu, dt 2 fs, friction 0.01/fs, frames every 20 fs after 0.1 ps
(E, F, stress, mol_id kept; saved frames carry normal masses - labels do not depend on masses)."""
import sys
import jax

jax.devices()
import numpy as np  # noqa: E402
from ase import units  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from ase.optimize import FIRE  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

start, out, T, ps = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]); seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
a = read(start); a.calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-m", stress=True, bucket_strategy="powers_of_2")
FIRE(a, logfile=None, maxstep=0.1).run(fmax=0.5, steps=100)
m = a.get_masses(); m0 = m.copy(); m[a.numbers == 1] = 4.0; a.set_masses(m)
MaxwellBoltzmannDistribution(a, temperature_K=T, rng=np.random.default_rng(int(T) + 1000 * seed))
dyn = Langevin(a, 2.0 * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=np.random.default_rng(1 + seed))
open(out + ".tmp", "w").close()
for k in range(int(ps * 1000 / 20)):
    dyn.run(10)
    if (k + 1) * 20 >= 100:
        b = a.copy(); b.set_masses(m0); b.calc = SinglePointCalculator(b, energy=a.get_potential_energy(), forces=a.get_forces(), stress=a.get_stress())
        b.info.update(T=T, t_ps=(k + 1) * 0.02, src="teacher_md"); write(out + ".tmp", b, format="extxyz", append=True)
        print(f"t {(k + 1) * 0.02:.2f} ps T {a.get_temperature():.0f} K", flush=True)
import os  # noqa: E402
os.replace(out + ".tmp", out)
