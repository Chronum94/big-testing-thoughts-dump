import json, sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    print(d['a_bcc'], round(d['rho_gcc'], 3), round(d['V_cm3mol'], 3), round(d['dOO'], 3), round(d['P_relax_GPa'], 1), round(d['P_X_GPa'], 1),
          round(d['E_relax_per_mol'] - d['E_X_per_mol'], 4), round(d['asym'], 3), d['relax_steps'], round(d['dOH_near_mean'], 3))
