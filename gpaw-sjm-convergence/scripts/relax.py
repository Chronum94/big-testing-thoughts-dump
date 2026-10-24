"""Short constant-potential relaxation of the top Au layer (displaced outward)."""
import json
import sys

from ase.constraints import FixAtoms
from ase.optimize import BFGS
import bench_setup

tag = sys.argv[1]
extra = eval(sys.argv[2]) if len(sys.argv) > 2 else {}
atoms, calc = bench_setup.make(tag, extra, target=4.0, n0=0.0)
atoms.positions[-1, 2] += 0.15
atoms.set_constraint(FixAtoms(indices=[0, 1]))
atoms.calc = calc
BFGS(atoms, logfile=f'{tag}.opt').run(fmax=0.05, steps=5)
txt = open(f'{tag}.txt').read()
out = dict(tag=tag, extra=repr(extra), energy=atoms.get_potential_energy(),
           attempts=txt.count('Potential found'),
           scf_iters=sum(int(l.split()[2]) for l in txt.splitlines()
                         if l.startswith('Converged after')))
print(json.dumps(out))
open('relax.jsonl', 'a').write(json.dumps(out) + '\n')
