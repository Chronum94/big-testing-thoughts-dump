| run | sj extras | attempts | SCF iters | wall (s) | phi - phi* (V) | E - E_ref (meV) |
|---|---|---|---|---|---|---|
| est_Au40 | {} | 4 | 116 | 53 | -0.0069 | -0.24 |
| base_Au40 | {} | 6 | 144 | 64 | -0.0066 | -0.29 |
| est_Pt45 | {} | 4 | 125 | 91 | -0.0009 | -0.02 |
| base_Pt45 | {} | 5 | 190 | 134 | -0.0044 | -0.48 |
| base_Cu40 | {} | 6 | 199 | 123 | -0.0058 | - |
| est_Cu40 | {} | 6 | 167 | 108 | -0.0071 | - |
| inx1_Au40 | {'inexact_newton': 0.1} | 4 | 52 | 31 | -0.0023 | +0.07 |
| inx1_Pt45 | {'inexact_newton': 0.1} | 7 | 145 | 89 | +0.0078 | +0.48 |
| inx1_Cu40 | {'inexact_newton': 0.1} | 6 | 94 | 62 | +0.0069 | - |
| inx05_Au40 | {'inexact_newton': 0.05} | 4 | 67 | 35 | -0.0028 | -0.04 |
| inx05_Pt45 | {'inexact_newton': 0.05} | 5 | 149 | 85 | -0.0080 | -0.77 |
| inx05_Cu40 | {'inexact_newton': 0.05} | 6 | 114 | 77 | +0.0002 | - |
| loose_Au40 | {'tol': 0.1, 'energy_correction': True} | 2 | 46 | 26 | +0.0560 | +0.11 |
| looseinx_Au40 | {'tol': 0.1, 'energy_correction': True, 'inexact_newton': 0.1} | 3 | 36 | 21 | +0.0694 | -0.04 |
| loose_Pt45 | {'tol': 0.1, 'energy_correction': True} | 3 | 98 | 62 | -0.0127 | -0.16 |
| loose05_Au40 | {'tol': 0.05, 'energy_correction': True} | 3 | 64 | 34 | +0.0225 | -0.06 |
| looseinx_Pt45 | {'tol': 0.1, 'energy_correction': True, 'inexact_newton': 0.1} | 4 | 79 | 53 | +0.0074 | -0.11 |
| loose05_Pt45 | {'tol': 0.05, 'energy_correction': True} | 3 | 98 | 58 | -0.0127 | -0.16 |
| tight0_Au40 | {'inexact_newton': 0.1} | 1 | 27 | 19 | +0.0029 | +0.05 |
| tight0b_Au40 | {'inexact_newton': 0.1, 'tol': 0.05} | 2 | 28 | 21 | +0.0030 | +0.03 |
