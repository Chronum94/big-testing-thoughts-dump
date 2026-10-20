"""Teacher (PET-MAD-m) Langevin NVT on 50-atom LGPS cells: held-out MD test frames.

usage: python md_teacher.py relaxed_all.extxyz IDX T_K PS out.extxyz
"""
import sys
import time

import jax

jax.devices()
import numpy as np  # noqa: E402
from ase import units  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import read, write  # noqa: E402
from ase.md.langevin import Langevin  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

a = read(sys.argv[1], int(sys.argv[2])); T, ps = float(sys.argv[3]), float(sys.argv[4])
a.calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-m", stress=True)
MaxwellBoltzmannDistribution(a, temperature_K=T, rng=np.random.default_rng(int(T)))
dyn = Langevin(a, 0.5 * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=np.random.default_rng(0))
out, t0 = [], time.time()


def grab():
    if dyn.nsteps < 2000:  # skip the first 1 ps
        return
    b = a.copy()
    b.calc = SinglePointCalculator(b, energy=a.get_potential_energy(), forces=a.get_forces(), stress=a.get_stress())
    b.info.update(category=f"test_md_{int(T)}K", ge=a.info.get("ge"), li_id=a.info.get("li_id"), T=T)
    out.append(b)


dyn.attach(grab, interval=50)  # every 25 fs
dyn.run(int(ps * 2000))
write(sys.argv[5], out)
d = a.get_all_distances(mic=True); np.fill_diagonal(d, 9)
print(f"{sys.argv[2]} {T} K: {len(out)} frames, {time.time() - t0:.0f}s, final T {a.get_temperature():.0f} K, min d {d.min():.2f}", flush=True)
