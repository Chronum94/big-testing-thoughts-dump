"""SJM convergence benchmark: fcc(111) 1x1x3 slab, constant potential.

usage: python bench.py TAG TARGET N0 [key=value ...]
keys 'element', 'unsolv_backside', 'keep_mixer' go to the setup, the rest to sj.
"""
import json
import sys
import time

import bench_setup

tag, target, n0 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
extra, setup = {}, {}
for kv in sys.argv[4:]:
    k, v = kv.split('=', 1)
    try:
        v = eval(v)
    except NameError:
        pass  # plain string
    (setup if k in ['element', 'unsolv_backside', 'keep_mixer', 'adaptive'] else extra)[k] = v

atoms, calc = bench_setup.make(tag, extra, target, n0, **setup)
t0 = time.time()
e = atoms.get_potential_energy()
txt = open(f'{tag}.txt').read()
out = dict(tag=tag, target=target, n0=n0, setup=repr(setup),
           extra=repr(extra), energy=e,
           ne=calc.parameters.sj.excess_electrons,
           phi=calc.get_electrode_potential(),
           attempts=txt.count('Potential found'),
           scf_iters=sum(int(l.split()[2]) for l in txt.splitlines()
                         if l.startswith('Converged after')),
           seconds=time.time() - t0)
print(json.dumps(out))
with open('results.jsonl', 'a') as f:
    f.write(json.dumps(out) + '\n')
