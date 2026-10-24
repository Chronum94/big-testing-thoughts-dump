"""NVE energy conservation per run: total energy (potential + kinetic) per atom vs time, linear drift and fluctuation.
usage: python nve_drift.py RUN_DIR [RUN_DIR ...]   (reads RUN_DIR/md.h5 written by apax MD; ASE units)"""
import sys
import h5py
import numpy as np
from ase import units
from ase.io import read

FRAME_FS = 20 * 0.5  # sampling_rate x dt of the md_*.yaml configs
m = read("/home/chronum/scratch/ecemc/md_radial/start_7722.extxyz").get_masses()
print(f"{'run':16s} {'frames':>6s} {'T start/end (K)':>16s} {'drift (meV/atom/ps)':>20s} {'E_tot std (meV/atom)':>21s} {'max |E-E0| (meV/atom)':>22s}")
for d in sys.argv[1:]:
    f = h5py.File(f"{d}/md.h5", "r"); n = len(f["observables/atoms/potential_energy/step"])
    # apax's h5 "time" is sample index x dt (not step x dt), so build time from the sampling interval
    t = np.arange(1, n + 1) * FRAME_FS / 1000.0  # ps
    pe = np.asarray(f["observables/atoms/potential_energy/value"][:n]); v = np.asarray(f["particles/atoms/velocity/value"][:n])
    ke = 0.5 * np.einsum("i,fij->f", m, v ** 2); N = len(m)
    T = 2 * ke / (3 * N * units.kB); e = (pe + ke) / N * 1e3  # meV/atom
    slope = np.polyfit(t, e, 1)[0]
    print(f"{d.rstrip('/'):16s} {n:6d} {T[0]:7.0f} / {T[-1]:6.0f} {slope:20.4f} {e.std():21.4f} {np.abs(e - e[0]).max():22.4f}")
