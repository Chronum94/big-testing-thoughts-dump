"""Relabel extxyz frames with the PET-MAD-m teacher (keeps student_energy / student_forces). usage: python label_extxyz.py OUT IN [IN ...]"""
import sys

import jax

jax.devices()
import numpy as np  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import read, write  # noqa: E402
from petjax import UPETCalculator  # noqa: E402

calc = UPETCalculator.from_checkpoint("/home/chronum/scratch/petjax/checkpoints/pet-mad-m", stress=True, bucket_strategy="powers_of_2")
import os, time
# resumable: frames already in OUT are skipped (a host GPU-driver crash can hang any call; the guard kills and retries)
done = len(read(sys.argv[1], ":")) if os.path.exists(sys.argv[1]) and os.path.getsize(sys.argv[1]) else 0
n, t0 = 0, time.time()
for p in sys.argv[2:]:
    try:
        fr = read(p, ":")
    except Exception as e:
        print("skip", p, e); continue
    for a in fr:
        n += 1
        if n <= done:
            continue
        a.calc = calc
        e, f, s = a.get_potential_energy(), a.get_forces(), a.get_stress()
        a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s); write(sys.argv[1], a, append=True)  # written as we go, so a timeout keeps what is done
    print(p, len(fr), f"{time.time() - t0:.0f}s", flush=True)
print("total", n)
