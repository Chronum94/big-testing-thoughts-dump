# Superionic ice (H2O) bootstrap plan: literature check + teacher credibility

Date: 2026-10-20. Teacher: PET-MAD v1.6 through pet-jax (CPU, float32 unless stated).
Detailed literature tables with full provenance: `lit_phase.md` (phase diagram, D_H, melting, protocols) and `lit_eos_teacher.md` (300 K EOS, symmetrisation, training-data coverage).
"Ver" means: **V** = the number was read in the primary paper, its open-access manuscript, a data repository or a full-text abstract. **S** = read in a secondary source quoting the paper (primary paywalled). **D** = derived by us (fit evaluation, interpolation, teacher calculation). Blocked or paywalled URLs are in `BLOCKED.txt`.

## 0. Bottom line

- **Static EOS, 15–200 GPa:** both pet-mad-s and pet-mad-m reproduce the experimental 300 K ice VII/X P(V) to within about ±5% (Loubeyre 1999, Hemley 1987 up to ~100 GPa, Brown & Journaux 2020). Below ~10 GPa the static teacher pressure is 2–3 GPa low, which is the expected size of the 300 K thermal pressure plus zero-point pressure, so it is not counted as an error.
- **Above 200 GPa the two models diverge.**
  - pet-mad-m stays within about 2% of the Brown 2020 BM4 fit up to 350 GPa; the fit is DFT-based above 150 GPa.
  - pet-mad-s is 6% low at 260 GPa, 11% low at 320 GPa and 17% low at 390 GPa.
- **H-bond symmetrisation (static, classical H):** the double well vanishes at d_OO ≈ 2.25 Å, which is ρ ≈ 3.4 g/cm³ and ~130 GPa, for both s and m. That is somewhat later than static PBE (116 GPa at 0 K; Cherubini 2024, S). This direction is expected for r2SCAN, which has a larger barrier than PBE. The experimental and NQE value of ~60–65 GPa is not directly comparable because it includes quantum effects. At 2000–5000 K the barrier (≤0.2 eV per H2O for P ≥ 50 GPa) is below k_BT·few, so this detail matters little for superionic MD.
- **Teacher MD probe (bcc, 54 H2O, ρ = 3.20 g/cm³, ~2830 K, 0.6 ps):**
  - The O lattice stays solid and the H diffuses, with D_H = 5.4e-4 cm²/s at P_tot = 140 GPa.
  - That is the same order as Cheng 2021's PBE-MLP bcc values (6.98e-4 at 100 GPa and 7.33e-4 at 200 GPa, both at 3000 K).
  - It also matches the Millot 2019 bcc density–pressure points (3.07 g/cm³ at 101 GPa, 3.35 g/cm³ at 161 GPa, ~1950 K).
  - At 1500 K and the same density, the H stays bonded (MSD_H flat at 0.11 Å², P_tot 118 GPa). This matches Cheng's classical bcc Ts of ~2370 K at 150 GPa: in the teacher, the superionic transition lies between 1500 and 2830 K.
- **Training data coverage:** MAD-1.6 and OMat24 contain no compressed/high-pressure H–O data (§3). The good EOS is therefore extrapolation that happens to work, not a guarantee.
- **The teacher is not exactly rotation invariant.** PET is not equivariant by construction, and the error grows with compression:
  - For s, the energy spread over 12 random rigid rotations is 1.6–3.4 meV/atom and the force noise is 0.007–0.09 eV/Å.
  - For m, the spread is 0.5–1.5 meV/atom and the force noise 0.004–0.035 eV/Å.
  - Perfect symmetric ice X has non-zero forces even though they should vanish by site symmetry: 0.02–0.09 eV/Å (s) and 0.01–0.07 eV/Å (m). This does not depend on cell size, float64 or the neighbour count.
  - With s, relaxing "below" symmetric ice X at >160 GPa gains 8–51 meV/H2O at a near-symmetric geometry, which is an artifact. With m the gain is ≤0.2 meV.
- **Teacher recommendation:**
  - Label with **pet-mad-m**, at ~3x the cost of s; it is mandatory for ρ ≥ 3.6 g/cm³ (P ≳ 170 GPa).
  - pet-mad-s is acceptable for ρ ≤ 3.4 g/cm³.
  - Treat ~1 meV/atom and ~0.03 eV/Å as the irreducible teacher noise floor for the (equivariant) student.
  - Do not trust any teacher prediction above ~350–400 GPa: the EOS has no data there and the noise grows.

## 1. Phase diagram (superionic bcc/fcc, O-sublattice melting, SI onset)

| Quantity | Value | Conditions | Source DOI | Ver |
|---|---|---|---|---|
| bcc O lattice (shock + XRD) | ρ = 3.072, 3.349, 3.547, 3.707, 3.883 g/cm³ | 101, 161, 195, 227, 249 GPa at 1920, 1970, 2230, 2250, 1670 K (±100–390 K) | 10.1038/s41586-019-1114-6 | V (Suppl. Table 2) |
| fcc ice XVIII (SI) | ρ = 3.890, 4.164, 4.546, 3.419, 4.097 g/cm³ | 278, 309, 420, 164, 323 GPa at 2340, 2380, 2680, 3230, 3800 K | 10.1038/s41586-019-1114-6 | V |
| bcc→fcc (shock) | 249–278 GPa | ~2000 K | 10.1038/s41586-019-1114-6 | V |
| SI conductivity / melt on ice VII Hugoniot | SI 100–190 GPa, melting 4800 K at 190 GPa | precompressed shock | 10.1038/s41567-017-0017-4 | V |
| bcc-SI onset / fcc-SI onset (DAC) | ≥20 GPa, ≥900 K / ≥29 GPa, ≥1300 K; only fcc-SI above 60 GPa | laser-heated DAC | 10.1038/s41567-021-01351-8 | V (abstract) |
| bcc→fcc line (DAC) | 57 GPa/1500 K → 166 GPa/2500 K | DAC XRD | 10.1103/PhysRevLett.128.165701 | V (abstract); disputed by Prakapenka et al. |
| VII–SI–liquid triple point | 14.6 GPa, 850 K | DAC XRD | 10.1103/PhysRevLett.125.195501 | V (abstract) |
| Mixed close-packed SI | >150 GPa, ~2500 K; bcc-SI at 25–50 GPa, 900–1300 K, ρ 2.15–2.37 | multishock XFEL | 10.1038/s41467-025-67063-2 | V |
| bcc→fcc (DFT-TI) | 100 ± 50 GPa | 3000 K | 10.1103/PhysRevLett.110.151102 | V |
| bcc stable (MLP, PBE + NQE) | ≲75 GPa, ≲2500 K; hcp ≥ fcc at ≳400 GPa | free energies | 10.1038/s41567-021-01334-9 | V |
| Tm (O melting), bcc, MLP | 1834 / 2592 / 3190 / 4126 / 4827 K | 50 / 75 / 100 / 150 / 200 GPa | Cheng 2021 data repo (as above) | V/D (interpolated) |
| Tm, fcc, MLP | 1858 / 2601 / 3246 / 4236 / 4832 K | 50 / 75 / 100 / 150 / 200 GPa | same | V/D |
| Ts (D_H = 1e-4 cm²/s), bcc, MLP | 2112 / 2371 / 2469 / 2442 / 2353 K | 100 / 150 / 200 / 300 / 400 GPa | same | V/D |
| Ts, fcc, MLP | 1490 / 1683 / 1789 / 1918 / 1935 K | 100 / 150 / 200 / 300 / 400 GPa | same | V/D |
| Tm (exp.) | 4800 K at 190 GPa (shock); partial melt 5200 K at 150 GPa (DAC); ≥3800 K at 323 GPa | | Millot 2018; Prakapenka 2021; Millot 2019 | V |
| SI onset (AIMD) | ~1500 K at 50 GPa; ~2000 K at 100–200 GPa | PBE | 10.1038/ncomms9156 | V |

Experiments put the superionic onset much lower than the classical PBE MLP does (900–1300 K vs ~2000 K). Classical MD with any PBE-like teacher should match the **simulation** Ts and Tm values, not the DAC onsets.

## 2. Reference numbers for MD

| Quantity | Value | Conditions | Source | Ver |
|---|---|---|---|---|
| D_H, bcc | 3.91e-5 / 6.98e-4 cm²/s | 100 GPa, 2000 / 3000 K | Cheng 2021 repo D-bcc.dat (MLP of PBE, classical) | V |
| D_H, bcc | 2.65e-6 / 1.38e-4 / 7.33e-4 cm²/s | 200 GPa, 2000 / 2500 / 3000 K | same | V |
| D_H, fcc | 3.57e-4 / 7.22e-4 cm²/s | 100 GPa, 2000 / 3000 K | same | V |
| D_H, fcc | 2.96e-4 / 7.40e-4 / 1.23e-3 cm²/s | 200 GPa, 2000 / 3000 / 4000 K | same | V |
| D_H, fcc | 1.42e-4 / 6.93e-4 / 1.23e-3 / 1.74e-3 cm²/s | 300 GPa, 2000–5000 K | same | V |
| D_H / D_O, liquid | 1.22e-3 / 1.14e-4 cm²/s | 100 GPa, 4000 K | same | V |
| D_O in the SI phase | ≈0 | while the O lattice holds | same | V |
| D_H, AIMD | 3.6e-4 cm²/s | 2000 K, low compression; <1e-4 at higher compression | 10.1103/PhysRevLett.117.135503 | V |
| Conductivity | ~90 S/cm (bcc D2O, 200 GPa, 2500 K, AIMD); ~150 S/cm (shock) | | Sun 2015; Millot 2018 | V |
| **Teacher (this work)** D_H | 5.4e-4 cm²/s, D_O ≈ 0 (MSD_O plateau 0.15 Å²) | bcc 54 H2O, ρ 3.20, 2830 K, P_tot 140 GPa, 0.6 ps NVT | `md/s_rho3.2_T3000` | D |

Simulation protocols (all V unless noted):

| Paper | Functional | Cell | dt | Length | Ensemble | NQE |
|---|---|---|---|---|---|---|
| Cheng 2021 | PBE (VASP 1000 eV) → NNP, cutoff 6.35 Å | 384 at. bcc / 324 fcc; 20,736 at. coexistence | 0.25 fs | 50 ps; 75 ps coexistence | NPT Nosé-Hoover (20 / 200 fs) | PIMD, 32 beads, as a correction |
| Millot 2018 / 2019 DFT | PBE, 900 / 1000 eV, Γ point | 162 / 324 atoms | 0.4 fs | – | NVT | ZPE correction only |
| Schwegler 2008 | PBE (Qbox) | 128 / 256 H2O | 0.24 fs | 5 + 5 ps | NPT | classical |
| French 2009 | PBE, Γ point | 54 H2O | – | 0.5–20 ps | NVT | classical |
| Sun 2015 | PBE (QE) | ≥108 D2O | – | ≥1 ps per step | NPT | classical |
| Hernandez & Caracas 2016 | PBE | 384 atoms | – | 1–2 + 2–8 ps | NVT | classical |

## 3. Static ice VII/X EOS (300 K) and symmetrisation

| Fit | Form | V0 (cm³/mol) | K0 (GPa) | K0' | Range (GPa) | DOI | Ver |
|---|---|---|---|---|---|---|---|
| Hemley 1987 | BM3 | 12.3 | 23.7 | 4.15 | 4.3–128 | 10.1038/330737a0 | S |
| Loubeyre 1999 | Vinet | 14.52 | 4.26 | 7.75 | 2.2–170 | 10.1038/17300 | S |
| Frank, Fei, Hu 2004 | BM3 | 12.4 | 21.1 | 4.4 | 6.6–60.5 | 10.1016/j.gca.2003.12.007 | V |
| Grande 2022, ice X | Vinet | 10.18 (33.82 Å³/2 H2O) | 50.5 | 4.50 | 30.9–88 | 10.1103/PhysRevB.105.104109 | V |
| Brown & Journaux 2020 | BM4, K'' −0.09 | 12.7218 | 19.2 | 3.8 | to 450 (DFT above 150) | 10.3390/min10020092 | V |
| Measured V (Fukui 2021) | 9.55 / 7.40 / 6.49 cm³/mol | at 10 / 35 / 60 GPa | arXiv:2104.11107 | V |

Fits give V = 6.64–6.78 cm³/mol at 50 GPa, 5.57–5.74 at 100 GPa and 4.93–5.17 at 150 GPa (D).

Symmetrisation:
- Experiment:
  - ~60 GPa by IR in H2O (Goncharov 1996, Aoki 1996: 62 GPa; V, abstracts).
  - 75 GPa by NMR (Meier 2018, Nat. Commun. 9, 2766; V).
  - Neutron D2O data become unimodal at ~80 GPa (Komatsu 2024; V).
- Theory:
  - Static PBE: 116 GPa at 0 K, 80 GPa at 300 K (Cherubini 2024; S).
  - With NQE: 60.6 GPa for SCAN, at d_OO ≈ 2.38 Å (Zhang & Wang 2025; V); 61–72 GPa for various GGAs.

### Teacher EOS check (this work)

Setup:
- Cell: 16 H2O, a 2×2×2 supercell of the 2-molecule Cu2O-type bcc cell. Pressure is the ASE stress, −tr(σ)/3, converted from eV/Å³ to GPa.
- Starting H arrangement: ordered ice-rule pattern (2 H per O), O–H = 1.0 Å, 0.02 Å rattle.
- Relaxation: FIRE, fmax 0.005 eV/Å, positions only at fixed cell.
- Also computed: symmetric ice X single point and a coherent off-centring scan E(δ).

Files:
- Scripts: `teacher_eos.py`, `plot_eos.py`, `symcheck.py`, `rotcheck.py`.
- Data: `eos/pet-mad-{s,m}.jsonl`, `eos/compare.txt`, `eos/rotcheck.txt`.
- Plot: `teacher_eos.png`.

| a (Å, 2-mol cell) | ρ (g/cm³) | V (cm³/mol) | d_OO (Å) | P s (GPa) | P m (GPa) | P Loubeyre (GPa) | P Brown (GPa) | Well depth s / m (meV/H2O) |
|---|---|---|---|---|---|---|---|---|
| 3.20 | 1.826 | 9.87 | 2.771 | 5.1 | 5.0 | 6.8 | 8.1 | 1391 / 1375 |
| 3.00 | 2.216 | 8.13 | 2.598 | 20.6 | 19.8 | 19.6 | 21.9 | 691 / 662 |
| 2.90 | 2.453 | 7.34 | 2.511 | 34.4 | – | 32.0 | 34.4 | 412 / – |
| 2.80 | 2.725 | 6.61 | 2.425 | 53.9 | 53.6 | 51.5 | 53.6 | 193 / 187 |
| 2.70 | 3.040 | 5.93 | 2.338 | 82.5 | – | 81.9 | 84.1 | 50 / – |
| 2.65 | 3.215 | 5.60 | 2.295 | 104.5 | – | 102.9 | 105.8 | 18 / – |
| 2.60 | 3.404 | 5.29 | 2.252 | 132.6 | 129.9 | 129.1 | 133.7 | 1.3 / 0 |
| 2.50 | 3.829 | 4.71 | 2.165 | 209.9 | 220.0 | 202.1 | 216.6 | 0 / 0 |
| 2.40 | 4.328 | 4.16 | 2.078 | 320.1 | 354.2 | 315.1 (extrap.) | 359.2 | 0 / 0 |
| 2.30 | 4.917 | 3.66 | 1.992 | 467.8 | 544.8 | 489.6 (extrap.) | 611.1 | 0 / 0 |

The comparison is static teacher vs 300 K experiment. At ≥20 GPa the 300 K thermal pressure is ≲1–2 GPa, so the differences above are mostly real.

Rotation test (`rotcheck.py`; 12 random rigid rotations; 16 H2O; columns for s / m):

| Structure | E spread (meV/atom) | Force dev. RMS (eV/Å) |
|---|---|---|
| a 3.0, ice VII | 1.6 / 1.5 | 0.007 / 0.004 |
| a 2.6, rattle 0.15 | 1.9 / 0.9 | 0.029 / 0.021 |
| a 2.4, rattle 0.15 | 3.4 / 1.5 | 0.087 / 0.035 |

## 4. Teacher MD probe

Setup: pet-mad-s, bcc 54 H2O (3×3×3 of the 2-mol cell), Langevin NVT (friction 0.01/fs), dt 0.25 fs, 2400 steps (0.6 ps). Script `teacher_md.py`; analysis `ana_md.py`.

| ρ (g/cm³) | T (K) | P_tot (GPa) | D_H (cm²/s) | D_O | MSD_O at end (Å²) | Force comp. RMS / max (eV/Å) | Min pair distance (Å) |
|---|---|---|---|---|---|---|---|
| 3.20 | 2830 (target 3000) | 139.9 | 5.4e-4 | ~0 (6e-6 from vibrational drift) | 0.15 | 2.3 / 11.8 | 0.81 |
| 3.20 | 1483 (target 1500) | 117.9 | <1e-6 (MSD_H flat at 0.11 Å²) | ~0 | 0.04 | 1.7 / 9.6 | 0.87 |

Cross-check: Millot 2019 bcc states interpolate to ~130 GPa at 3.20 g/cm³ and ~1950 K (D). Adding ~10 GPa of extra thermal pressure for 2830 K is consistent with the teacher's 140 GPa. This is a consistency check, not a precise test, because the experimental temperature uncertainty is ±100–390 K.

## 5. Bootstrap training blocks (no MD frames)

General rules:
- Label everything with pet-mad-m; s is allowed only for ρ ≤ 3.4.
- Drop frames with max|F| > 20 eV/Å or any O–H < 0.80 Å, H–H < 0.70 Å or O–O < 1.85 Å. Teacher MD at 2830 K reached max|F| 11.8 eV/Å, RMS 2.3 eV/Å and a minimum pair distance of 0.81 Å.
- **Rattles must stay small.** A 0.15 Å isotropic Gaussian rattle of all atoms gave a force RMS of 13–23 eV/Å and a max of 58–122 eV/Å, 6–10× the thermal forces at 2830 K. Bond-stretch energy dominates, so isotropic H rattles above ~0.08 Å produce off-manifold frames the student does not need.

Densities (static teacher P in brackets; thermal pressure at 2000–4000 K adds roughly 10–25 GPa):
- ρ = 2.2 [20 GPa], 2.5 [37], 2.8 [60], 3.1 [90], 3.4 [130], 3.7 [180], 4.0 [245], 4.3 [315] g/cm³.
- This spans 25–350 GPa and covers every Cheng/Millot reference state.
- Do not go above 4.6 g/cm³ (~390 GPa static): the EOS is untested there and the teacher noise grows.

| Block | Cells | Densities | H configurations | Perturbation | Frames | Why |
|---|---|---|---|---|---|---|
| A ice VII/X lattice | bcc 54 H2O, fcc 32 H2O | all 8 | ice-rule disordered (random 2-in/2-out, ≥4 seeds) plus symmetric X | O σ 0.03–0.10 Å; H σ 0.03–0.08 Å; ±3% isotropic strain | 8 × 2 × 25 = 400 | EOS, phonons and the double well near the stable manifold |
| B superionic proxy | bcc 54, fcc 32 | 2.8–4.3 | H on random bond-midpoint, tetrahedral or octahedral interstitials; random occupation, so ionic defects (H3O+/OH−-like) are allowed | O σ 0.08–0.2 Å (the teacher 2830 K O MSD of 0.15 Å² is about 0.22 Å RMS per component); H σ 0.05 Å after placement; min-distance filter | 6 × 2 × 40 = 480 | Broken and re-forming O–H bonds; the key superionic configurations |
| C fully random H | bcc 54, fcc 32 | 3.1–4.3 | H uniform-random with O–H ≥ 0.85 Å and H–H ≥ 0.75 Å | O σ 0.1–0.2 Å | 150 | Liquid-like H sublattice; tail coverage |
| D volume / strain scans | 16 H2O bcc + fcc | 1.8–4.6, 20 points | relaxed ice VII/X | ±6% volume, ±4% shear | 2 × 20 × 3 = 120 | Stress labels; P(V) for NPT |
| E Bain path bcc↔fcc | 54 H2O (bct) | 3.4, 3.8, 4.3 | ice-rule and random H | c/a 1 → √2 in 8 steps, H as in B | 3 × 8 × 4 = 96 | bcc/fcc relative energy (transition ~100–280 GPa) |
| F O-disordered (liquid-like) | 54 H2O | 3.0, 3.6, 4.2 | random molecular and dissociated | random O with O–O ≥ 1.9 Å, then H as C | 60 | Liquid side of Tm (D_O ~1e-4 at 4000 K/100 GPa) |
| G close-packed alternative | hcp 48 H2O (optional) | 3.8–4.3 | as in B | as in B | 40 | hcp ≥ fcc reported at ≳400 GPa (Cheng); close-packed mixtures >150 GPa (Andriambariarijaona 2025) |

**Afterwards:** run one round of student MD at the test conditions below, select the high-uncertainty frames, relabel them with the teacher and add them back. Static blocks alone will not cover the correlated hopping.

## 6. Test MD (student) and observables

Cells: bcc 128 H2O (4×4×4, 384 atoms) and fcc 108 H2O (3×3×3 conventional, 324 atoms). These are Cheng's sizes.

Run settings:
- dt 0.25 fs throughout (Cheng uses 0.25 fs). The 0.4 fs used by Millot's AIMD is acceptable only up to ~3000 K, and only if the NVE drift is below 1 meV/atom/ps.
- Equilibrate NVT for 2 ps, then run NVT ≥ 20 ps for D_H, or NPT at the target P for density.

| Run | Lattice | ρ (g/cm³) / P (GPa) | T (K) | Expected result (reference) |
|---|---|---|---|---|
| T1 | bcc | 3.20 (~130–140 GPa) | 1500 | Not superionic: D_H ≲ 1e-5 (Ts bcc 2371 K at 150 GPa, Cheng; classical) |
| T2 | bcc | 3.20 | 2500 | Borderline SI: D_H ~1e-4 (Ts ~2370–2470 K) |
| T3 | bcc | 3.20 | 3000 | SI: D_H ≈ 7e-4 (Cheng bcc 3000 K); teacher 5.4e-4 at 2830 K |
| T4 | fcc | 3.9 (~250–280 GPa) | 3000 | fcc SI: D_H ≈ 7e-4 (Cheng 200–300 GPa); ρ 3.89 at 278 GPa/2340 K (Millot 2019) |
| T5 | bcc | 3.07 / 101 GPa (NPT) | 1950 | Density check vs Millot: 3.072 g/cm³ at 101 GPa, 1920 K |
| T6 | fcc | NPT 278 GPa | 2340 | 3.890 g/cm³ (Millot) |
| T7 | bcc → heat | 150 GPa (NPT), 3500 → 5000 K ramp | – | O melting: Tm ≈ 4126 K at 150 GPa (Cheng). A single-phase ramp superheats, so use coexistence (≥2000 atoms, 50–75 ps) for a real number |
| T8 | liquid | 3.0, 4000 K | – | D_H 1.2e-3, D_O 1.1e-4 at 100 GPa (Cheng) |

Observables, each scored as student vs teacher, with literature as a sanity check:
1. P(ρ, T) at T1–T4 vs teacher single points on 20 student frames.
2. D_H(T) at 3.20 g/cm³: Arrhenius slope and Ts where D_H = 1e-4 cm²/s.
3. D_O and the O Lindemann ratio, which must stay ≈0 below Tm.
4. g_OO, g_OH and g_HH, plus the O–H coordination-number histogram with a 1.3 Å cutoff (fraction of H2O, H3O+ and OH−).
5. Lifetime of O–H bonds: the H hop rate.
6. NVE energy drift at 0.25 and 0.4 fs.
7. Force and stress parity on held-out teacher frames, against the teacher noise floor of ~1 meV/atom and ~0.03 eV/Å.

## 7. Caveats

- The Cheng Ts and Tm values are classical PBE-MLP numbers interpolated from their repository. Experiments place the superionic onset lower: bcc-SI from ≥900 K at ≥20 GPa (Prakapenka 2021).
- The Wilson 2013 D table units look like a typo, so those numbers are not used.
- The French 2010 diffusion coefficients were not accessed.
- Weck 2022 is disputed.
- Some EOS fits were taken from secondary sources (marked S).
- The teacher MD covered bcc only, at 1500 and 2830 K, 0.6 ps each with 54 H2O; it was not tested on fcc, near Tm or in the liquid. These D values come from ~0.3 ps fits, so they are good to a factor of ~1.5.
- One check (`symcheck.py`) was mistakenly run on cores 12–15 while the m EOS ran on cores 8–11, i.e. briefly 8 threads. Everything else used `taskset -c 8-11`.
