"""Control: teacher C-N stretch in molecules (HCN, isolated [Fe(CN)6]4- not possible: neutral only;
use HCN and K4Fe(CN)6-like? -> HCN and CH3CN), relaxed in a 20 A box, normal modes via ASE.
Exp gas-phase fundamentals: HCN nu1(CN) 2097 cm-1, CH3CN nu2(CN) 2267 cm-1 (NIST CCCBDB; not re-verified here).
Usage: python hcn_control.py <model> <out.json>
"""
import os, sys, json
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from ase.build import molecule
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from petjax import UPETCalculator

model, out = sys.argv[1:3]
calc = UPETCalculator.from_checkpoint(f"/home/chronum/scratch/petjax/checkpoints/pet-mad-{model}", stress=False, default_dtype="float64")
res = {}
for name in ("HCN", "CH3CN"):
    m = molecule(name); m.cell = [20, 20, 20]; m.pbc = True; m.center(); m.calc = calc
    BFGS(m, logfile=None).run(fmax=0.005, steps=500)
    iC = [i for i, s in enumerate(m.get_chemical_symbols()) if s == "C"]
    iN = m.get_chemical_symbols().index("N")
    dCN = min(m.get_distance(i, iN) for i in iC)
    v = Vibrations(m, delta=0.005, name=f"/home/chronum/scratch/pba/lit/vibtmp/mol_{model}_{name}")
    v.run(); f = np.sort(v.get_frequencies().real); v.clean()
    res[name] = dict(d_CN=float(dCN), freqs_top=f[-4:].tolist())
    print(name, res[name], flush=True)
json.dump(res, open(out, "w"), indent=1)
