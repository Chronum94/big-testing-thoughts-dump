# Superionic H2O: literature numbers (compiled 2026-10-20)

Key: **V** = I read the number myself in the source (where it is given). **U** = unverified (from memory or a secondary source). Texts are in /home/chronum/scratch/untrusted/sice_lit/<name>/ (PDF to text via PyMuPDF).
Abbreviations: OSTI-AM = OSTI accepted manuscript; SI = superionic; Tm = melting temperature of the O sublattice; Ts = ice to SI transition temperature.

## Table A: phase-diagram points

| Phase | P (GPa) | T (K) | rho (g/cm3) | Method | DOI | Ver. | Where |
|---|---|---|---|---|---|---|---|
| bcc ice (likely insulating X) | 101±7 | 1920±320 | 3.072±0.054 | Reverberating shock + XRD (Omega) | 10.1038/s41586-019-1114-6 | V | Millot19 Suppl. Table 2 (OSTI-AM 1568026) |
| bcc | 161±12 | 1970±200 | 3.349±0.072 | same | same | V | same |
| bcc | 195±11 | 2230±280 | 3.547±0.063 | same | same | V | same |
| bcc | 227±11 | 2250±350 | 3.707±0.062 | same | same | V | same |
| bcc | 249±22 | 1670±100 | 3.883±0.104 | same | same | V | same |
| fcc ice XVIII (SI) | 278±13 | 2340±390 | 3.890±0.069 | same | same | V | same |
| fcc XVIII | 309±24 | 2380±370 | 4.164±0.108 | same | same | V | same |
| fcc XVIII | 420±24 | 2680±320 | 4.546±0.092 | same | same | V | same |
| fcc XVIII | 164±20 | 3230±480 | 3.419±0.138 | same | same | V | same |
| fcc XVIII | 323±40 | 3800±650 | 4.097±0.172 | same | same | V | same (lower bound on Tm) |
| bcc to fcc transition | 249-278 (at ~2000 K) | ~2000-3000 | d-spacing jump ~0.05 Å | same | same | V | main text |
| SI ice along ice-VII Hugoniot | 100-190 | 2000-5000 | — | Precompressed laser shock, VISAR/SOP | 10.1038/s41567-017-0017-4 | V | Millot18 main text (OSTI-AM 1542614) |
| Melting on ice-VII Hugoniot | 190 | 4800 | — | same (Us ~15 km/s) | same | V | main text |
| bcc-SI (ice XX) appears | ≥20 | ≥900 (900-1900 at 20-60 GPa) | — | LH-DAC XRD + optical | 10.1038/s41567-021-01351-8 | V | Prakapenka21 abstract (arXiv 2007.07715) and NSF PAR text |
| fcc-SI (XVIII) appears | ≥29 | ≥1300 | — | same | same | V | same |
| bcc-SI vanishes; only fcc-SI is stable | >60 | — | — | same | same | V | main text |
| bcc-SI / fcc-SI observed | 49 | 1600 / 2000 | — | same | same | V | Fig. 2 caption |
| fcc-SI, partially molten | 150 | 5200 | — | same | same | V | Fig. 2 caption |
| Conducting fluid (σ_opt >15 S/cm) | 17-105 | >4000 | — | Pulsed LH-DAC absorption | same | V | main text |
| Fluid to fcc-SI in Uranus (Neptune) | 56 (71) | isentrope | — | same | same | V | main text |
| bcc to fcc transition line | 57 → 166 | 1500 → 2500 | — | LH-DAC XRD | 10.1103/PhysRevLett.128.165701 | V | Weck22 abstract (HAL API) |
| bcc-SI stable | up to ≥100 | — | — | same | same | V | abstract |
| Triple point (VII'–VII''(bcc-SI)–liquid) | 14.6 | 850 | — | DAC XRD, ≤45 GPa/1600 K | 10.1103/PhysRevLett.125.195501 | V | Queyroux20 abstract (HAL API) |
| Mixed close-packed SI | >150 (max ~180) | ~2500 (2500-3000) | — | Multishock + XFEL XRD | 10.1038/s41467-025-67063-2 | V | Andriambariarijaona et al., Nat Commun (PMC12796318) |
| bcc + fcc signatures together | ~70-120 | 2000-2500 | — | same | same | V | main text |
| bcc-SI | ~25-50 | 900-1300 | 2.153(9), 2.207(2), 2.371(8) | same | same | V | main text |
| bcc to close-packed SI (AIMD) | 220-280 at 2200 K; 200-260 at 3000 K | — | — | NPT AIMD | 10.1038/ncomms9156 | V | Sun15 main text |
| bcc-SI to fcc-SI (DFT free energy) | 100±50 (1±0.5 Mbar) | 3000 (ΔG≈0 at 1 Mbar) | — | DFT-MD TI | 10.1103/PhysRevLett.110.151102 | V | Wilson13 arXiv 1211.6482 abstract, Table I |
| bcc stable (MLP) | ≲75 | ≲2500 | — | MLP + free energy | 10.1038/s41567-021-01334-9 | V | Cheng21 arXiv 2103.09035 text |
| hcp slightly more stable than fcc | ≳400 | (|Δμ| <10 meV/H2O over 100-800 GPa, 2000-5500 K) | — | same | same | V | text |
| VII'' (bcc-SI) to VII' first-order; ice X near 100 GPa | ~34 (at 1300 K) | 1300-2000 | — | AIMD | 10.1103/PhysRevLett.117.135503 | V | Hernandez-Caracas16, Fig. 2 caption (34.2/34.5 GPa) |
| Melting-slope kink | 47 | 1000 | — | Raman LH-DAC | 10.1103/PhysRevLett.94.125508 | V | Goncharov05 abstract (OSTI 1008685) |

## Table B: self-diffusion coefficients (1 Å²/ps = 1e-4 cm²/s)

| Phase | D_H | D_O | T (K) | P or ρ | Method | DOI | Ver. | Where |
|---|---|---|---|---|---|---|---|---|
| fcc-SI | 3.57e-4 cm²/s | ~0 | 2000 | 100 GPa | MLP(PBE) MD, classical | 10.1038/s41567-021-01334-9 | V | Cheng repo `diffusion-coefficient/D-fcc.dat` |
| fcc-SI | 7.22e-4 | ~0 | 3000 | 100 GPa | same | same | V | same |
| fcc-SI | 2.96e-4 / 7.40e-4 / 1.23e-3 | ~0 | 2000 / 3000 / 4000 | 200 GPa | same | same | V | same |
| fcc-SI | 1.42e-4 / 6.93e-4 / 1.23e-3 / 1.74e-3 | ~0 | 2000 / 3000 / 4000 / 5000 | 300 GPa | same | same | V | same |
| bcc | 3.91e-5 / 6.98e-4 | ~0 | 2000 / 3000 | 100 GPa | same | same | V | `D-bcc.dat` |
| bcc | 2.65e-6 / 1.38e-4 / 7.33e-4 / 1.23e-3 | ~0 | 2000 / 2500 / 3000 / 4000 | 200 GPa | same | same | V | same |
| liquid | 1.22e-3 | 1.14e-4 | 4000 | 100 GPa | same | same | V | `D-liquid.dat` |
| liquid | 1.21e-3 | 3.99e-5 | 4000 | 200 GPa | same | same | V | same |
| Ts criterion | D_H = 1e-4 cm²/s | — | — | — | definition | same | V | text |
| bcc-SI | 3.6e-8 m²/s (=3.6e-4 cm²/s), at low compression | 0 | 2000 | low-P end (≤~50 GPa) | AIMD PBE | 10.1103/PhysRevLett.117.135503 | V | main text |
| bcc-SI, higher compression | <1.0e-8 m²/s | 0 | 1300-2000 | higher P | same | same | V | same |
| bcc / fcc | 1.20 / 0.98 (2000 K); 6.49 / 4.23 (5000 K) | — | 2000, 5000 | 5 g/cm³ | DFT-MD | 10.1103/PhysRevLett.110.151102 | V* | Wilson13 Table II |
| bcc / fcc | 0.48 / 0.20 (2000 K); 2.48 / 1.33 (5000 K) | — | 2000, 5000 | 11 g/cm³ (~40 Mbar) | same | same | V* | same |
| Note: | the table states the units as "Å² fs⁻¹", which gives ~0.1 cm²/s and is implausible. Å²/ps (1e-4 cm²/s) matches the other studies, so it is probably a units typo. | | | | | | | |
| SI ice-VII, 50 GPa isobar | rises steeply 1300-1600 K, then more slowly | — | 1300-1600 | 50 GPa | FPMD PBE | 10.1073/pnas.0808137105 | V (qualitative; values only in a figure) | Schwegler08 OSTI-AM 945774 |
| bcc-SI (French 2010) | not accessed | | | | DFT-MD | 10.1103/PhysRevB.82.174108 | — | paywalled; no OA copy found |

Conductivity (related): Sun15 ionic σ ≈ 90 S/cm (bcc-SI D2O, 200 GPa, 2500 K) [V]. Millot18: reverberating-shock σ ≈ 150 S/cm vs ~30 S/cm on the liquid Hugoniot; σ_e(2 eV) ≪ 100 S/cm below 200 GPa and 5000 K [V]. Sugimura12 (JCP 137, 194505, 10.1063/1.4766816): >10 S/cm at 739 K/56 GPa and 749 K/62 GPa [V, from the abstract on an institutional repository; disputed as overestimated in arXiv 2401.11454].

## Table C: O-sublattice melting (Tm) and SI onset (Ts)

| Quantity | P (GPa) | T (K) | Source / DOI | Ver. | Where |
|---|---|---|---|---|---|
| Tm (experiment, shock) | 190 | 4800 (~5000) | Millot18 10.1038/s41567-017-0017-4 | V | text |
| Tm lower bound (XRD solid) | 323 | 3800±650 | Millot19 | V | Suppl. Table 2 |
| Tm fcc-SI, partial melt | 150 | 5200 | Prakapenka21 | V | Fig. 2 |
| Tm kink | 29 (42 for the VII–SI line) | — | Prakapenka21 | V | text |
| Tm bcc (MLP, PBE + NQE) | 50 / 75 / 100 / 150 / 200 | 1834 / 2592 / 3190 / 4126 / 4827 | Cheng21 repo `phase-boundary/Tm-bcc.dat` (linear interpolation) | V | data file |
| Tm fcc (MLP) | 50 / 75 / 100 / 150 / 200 | 1858 / 2601 / 3246 / 4236 / 4832 | `Tm-fcc.dat` | V | data file |
| Tm on coexistence | 100 | 3250 | Cheng21 | V | text (γ0 bcc 16.8±0.2, fcc 24.4±0.3 meV/Å²) |
| Ts bcc (MLP) | 100 / 150 / 200 / 300 / 400 | 2112 / 2371 / 2469 / 2442 / 2353 | `Ts-bcc.dat` | V | data file |
| Ts fcc (MLP) | 100 / 150 / 200 / 300 / 400 | 1490 / 1683 / 1789 / 1918 / 1935 | `Ts-fcc.dat` | V | data file (curve folds below ~75 GPa, so low-P interpolation is unreliable) |
| Ts (Cheng, general statement) | — | ~2000 | Cheng21 | V | text |
| Ts (H-sublattice melting) | 50 / 100-200 / >200 | ~1500 / ~2000 / bcc 2000 → CP 1700 → P21/c 1200 | Sun15 | V | text |
| Tm jump; SI diffusion below Tm | ~45 | Tm(43 GPa) ≈1600 K bisected | Schwegler08 | V | text |
| SI onset at 50 GPa (H diffuses, O fixed) | 50 | 1300-1400 | Schwegler08 | V | text |
| Onset, Hernandez-Caracas | ≤300 | SI regime from 1300 K; symmetric-bond SI at ≥1800 K | 10.1103/PhysRevLett.117.135503 | V | text |
| Cavazzoni99 SI phase | 30-300 GPa range studied | 300-7000 K studied; metallic water >7000 K | 10.1126/science.283.5398.44 | V (abstract only); SI onset ~2000 K **U** | Crossref abstract |

## Table D: simulation protocols

| Paper | Functional / code | Cell | dt | Run length | Ensemble / thermostat | NQE |
|---|---|---|---|---|---|---|
| Cheng21 MLP (RuNNer/n2p2 NNP, cutoff 6.35 Å) | PBE (VASP, ENCUT 1000 eV, POTIM 0.25 fs, Fermi smearing σ=0.2585 eV) [V, repo INCAR] | MD examples: bcc 384, fcc 324 atoms [V]; coexistence 20,736 atoms, 75 ps [V text] | 0.25 fs [V, npt.lmp] | example 200,000 steps (50 ps) [V] | LAMMPS NPT Nosé-Hoover, Tdamp 20 fs, Pdamp 200 fs [V] | PIMD with i-PI: 32 beads, 0.25 fs, 150k steps, PILE-G τ=10 fs (+GLE in the input) [V]; plus MLP→PBE free-energy perturbation |
| Millot18 DFT-MD | PBE PAW (VASP), 900 eV, Γ point | 162 atoms | 0.4 fs | — | NVT Nosé-Hoover | proton vibrational-energy correction to the EOS [V] |
| Millot19 DFT-MD (elastic constants) | PBE PAW (VASP), 1000 eV, Γ point | 324 atoms | 0.4 fs | — | NVT Nosé-Hoover | none stated [V] |
| Sun15 | PBE, norm-conserving pseudopotentials, Quantum ESPRESSO, 1500 eV constant-cutoff | ≥108 molecules | not in main text | ≥1 ps per P step (≥2 ps above 1.3 TPa) | NPT Parrinello-Rahman + Nosé-Hoover; D2O used to allow a larger dt | classical [V] |
| Wilson13 | VASP DFT-MD, two-step coupling-constant TI | in the SI (not accessed); notes that the earlier work used 54 molecules | — | — | constant cell | classical [V for VASP/TI; other details U] |
| French09 | PBE PAW (VASP 4.6), 900 eV, Γ point | mostly 54 molecules (16/128 tested) | not found | 0.5-2 ps EOS; up to 20 ps per direction at phase boundaries | NVT Nosé | classical [V] |
| Hernandez-Caracas16 | PBE PAW (VASP) | 128 H2O (4×4×4 bcc) = 384 atoms | not in main text | 1-2 ps thermalisation + 2-8 ps production | NVT isokinetic (velocity rescaling each step) | classical [V] |
| Schwegler08 | PBE, Qbox, norm-conserving pseudopotentials, 85 Ry | 128 molecules (single phase); 256 (two phase) | 10 a.u. (≈0.24 fs) | 5 ps equilibration + 5 ps; two-phase runs resolve in 5-6 ps | NPT Born-Oppenheimer MD | classical [V] |
| Goncharov05 MD | BLYP, CPMD 3.82, Troullier-Martins pseudopotentials | 54 molecules | 0.048 fs (Car-Parrinello) | 2 ps equilibration + 5-10 ps | fixed volume, 1500/2000 K; 2.0/2.2/2.4 g/cc ≈ 21/33/49 GPa | classical [V] |
| Cavazzoni99 | CPMD, constant P | not accessed (U: 54 or 64 molecules) | U | U | NPT | classical (U) |

Not accessed (blocked or paywalled): Cheng21 SI PDF (nature.com sign-in redirect), Weck22 and Queyroux20 full text (HAL bot check, so only abstracts via the HAL API), French10 PRB, Cavazzoni99 full text. The APS abstract pages returned 403.
