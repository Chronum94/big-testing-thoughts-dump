> **Note:** compiled with Claude (Anthropic's AI assistant) from three research sub-agents (published literature, code issue trackers and mailing lists, test-system design from our harness). No DFT has been run on any of these yet. Anything marked *inference* is the agents' or our extrapolation, not a reported fact.

# Hard constant-potential test cases for SJM

The easy cases are fcc(111) Au, Pt and Cu, 1×1×3 slabs, which converge in 3–6 outer steps. These candidates aim to break the assumptions the current scheme relies on:
- φ(N) is smooth, monotonic and nearly linear;
- the inner SCF is well behaved;
- a loosely converged φ is an unbiased measurement.

## Caveat on our benchmarks so far

The harness copied GPAW's `test_sjm.py` geometry: jellium top at 14.5 Å in a 15 Å cell. With the default bottom at top atom + 3 Å, the Au counter-charge is a sheet only **0.79 Å** thick. The capacitances and step statistics in `README.md` refer to that setup. A thicker, production-like jellium should be re-checked.

## Ranked candidates

The ranking is by stress per CPU-minute on 1 core. Costs are the design agent's estimates scaled from Au 1×1×3 (about 0.5 s per iteration).

### Tier 1: cheap, run first

1. **Au(111) 1×1×3, Γ-only, Fermi–Dirac 0.01 eV.** About 5–10 s per SCF.
   - Failure mode: a discrete spectrum, so C_q is a set of spikes and gaps and φ(N) is a staircase. The SCF sloshes between near-degenerate levels.
   - Real-world match: FCP-vasp-ase #3 (Cu, NELECT never converges; the Fermi level hops between k-point levels, fixed by more k-points or larger smearing).
   - Watch: sign flips of the regressed slope, attempts reaching max_iters, SCF iterations per attempt.

2. **Spill-out: negatively charged Pt(111), or Au 1×1×3 at 2.0 V with the jellium close** (`bottom = z_max + 1.5`). About 15–30 s per SCF.
   - Failure mode: "adding electrons is like raising E_F above the vacuum level". The electrons escape into the jellium well and solvent, so φ(N) saturates or has no root.
   - Real-world match: JDFTx #155 (charged Pt(111), even in vacuum); the comment in GPAW `sjm.py` about spill-out at low work function; the 7.0 V mirror case.
   - Watch: ∫n(z) beyond the jellium bottom, dφ/dN against its low-|N| value, max_step halving loops.

3. **Graphene 1×1 at the Dirac point** (6×6 Γ-centred k-points, FD 0.02 eV; an 18×18 control). About 0.5–1 min per SCF.
   - Failure mode: C_q → 0, so φ ∝ sign(N)√|N| and the slope is infinite at N = 0. The 6×6 mesh adds a fake gap.
   - Literature: the CIP paper expects larger E_F-based errors for low-DOS electrodes such as graphene.
   - Watch: overshoot of the first slope-estimate step (C_dl-dominated), outer attempts, comparison with the analytic √N.

4. **Low-density metals near the potential of zero charge (PZC): Mg(100) with 1 atom per layer, or Li(100).**
   - Failure mode: φ(N) is non-monotonic near the PZC (VASPsol #49: E_F of −2.42, −3.11, −2.60, −3.45 eV across a ΔN of 0.15). The cause is that the dielectric cavity reaches into the low-density metal (also JDFTx #313 for Na(100)). Li has the lowest work function, about 2.9 eV, so spill-out also applies (Hagopian et al., JCTC 2022).
   - Watch: φ(N) monotonicity, ε(z) inside the slab.

5. **Si(111)-H, `diamond111('Si',(1,1,4))` capped on both faces, cell 19 Å, 6×6 k-points.** About 1–2 min per SCF.
   - Failure mode: with E_F in the gap, N(φ) ∝ exp(±φ/σ), so φ(N) is logarithmic and dφ/dN ≈ −σ/N varies by orders of magnitude. Secant steps overshoot to the wrong band edge, and 4-point regression is meaningless. Deep in the gap there is effectively no root, since N falls below SCF noise.
   - Literature: Bhati et al., JPCC 2020 (3C-SiC: "a potential applied in the range of the bandgap is not sufficient to alter the occupancy of states"); Moradi & Melander, Curr Opin Electrochem 2025 ("the Fermi level cannot be continuously altered across the band gap region").
   - Targets: inside the gap, and just into each band.

6. **Elongated cell: the Au slab with a 30–40 Å solvent region.**
   - Failure mode: the CIP paper dropped the solvent for two-electrode cells "due to extreme convergence problems caused by the finite-difference (FD) Poisson solver… known to be unstable for highly elongated cells". FastPoissonSolver was incompatible with the dielectric and SJM as of 2024.
   - Watch: Poisson iterations, SJMDipoleCorrection iterations, SCF stalls.

### Tier 2: minutes per SCF, distinct physics

7. **MoS₂ monolayer basal plane** (`mx2('MoS2', vacuum=5)`, cell 16 Å, 6×6 k-points), with the target inside the 1.7 eV gap and 1.2 V into the conduction band. φ jumps across N = 0 with no root inside the gap. Tests whether the loop fails gracefully.
8. **Ni(111) 1×1×3, spin-polarised** (moments 0.7, FD 0.05), at 3.0, 4.5 and 6.0 V. The moment changes with N, giving kinks in C_q and possible hysteresis in φ(N). Related report: CP-VASP #23 (NiOOH with spin, endless oscillation of N and potential).
9. **O₂ flat on Au(111) 2×2×2, spin-polarised**, from 4.5 down to 3.0 V. As the narrow π* level crosses E_F, O₂ (moment 2) turns into O₂⁻ (moment 1): a near-discontinuous transfer of about 1 e. Expect a flat step in φ(N) and spin flipping. Related: GCE-cDFT for outer-sphere CO₂⁻ on Au (Qin et al., Nat Commun 2023), where E_F is set by the solute.
10. **Pt(111) 2×2×2 + explicit H₂O with ghost-atom cavity** (`H2O_layer={'style':'ghost_atoms','ghost_spacing':1.5}`), also with `jelliumregion={'bottom':'cavity_like'}`. A sharp cavity and a jellium sitting on the water's density tail lead to charge leakage. Pt's d-bands at E_F are themselves hard: JDFTx 2017 reports "deviations from exponential convergence" for negatively charged Pt(111).
11. **MgO(100), 3 layers** (cell 16 Å, 4×4 k-points). Wide gap with the conduction band near vacuum: added electrons may prefer the jellium or solvent over the slab.
12. **K on graphene 2×2.** A narrow K 4s resonance near the Dirac point gives partial, N-dependent charge transfer on top of a vanishing C_q.

### Tier 3: realistic and expensive

13. **Anatase TiO₂(101) PBE+U, 4–8 layers, 5×5 k-points** (Melander et al., npj Comput Mater 2024, the CIP paper). Gap pinning, polarons that depend on potential and give metastable SCF branches, and an inner potential not converged even at 8 layers. The authors' fix is CIP. Rutile TiO₂(110) 1×1 is a cheaper variant.
14. **CeO₂(111) with +U** (VASPsol #36, CP-VASP #6). Plain implicit solvation already fails. N ran away to 705 against a neutral count of 802, with absurd energies, because the inner SCF failed repeatedly.
15. **Al(111) 3×3×3 Γ-only, plus a 25 Å-cell variant.** Classic long-wavelength charge sloshing, re-excited by every kick in N.
16. **Fe–N–C single-atom sites.** Two stable Fe(II) spin states at ORR potentials, which needed occupation-matrix constraints (Zheng et al., JMCA 2023). A potential-driven S = 3/2 → 2 transition appears in constant-potential AIMD (Li et al., Chem Sci 2025). Abstract-level only.
17. **Outer-sphere redox: Ru(NH₃)₆³⁺ near Au(111) 5×5** (CIP paper). Solute states dominate E_F, so φ(N) is discontinuous and holding the oxidation state needs GCE-cDFT.
18. **Coupling to dynamics.** Constant-potential MD after equilibration fails to converge in 200 SCF steps (CP-VASP #25). Energy spikes at every N update (#11). Forces converge before the potential, leaving ±0.02 V fluctuation in relaxations (TPOT #10). Explicit-water AIMD at constant Fermi level "may be difficult to converge" (Andersson & Zhang review, arXiv:2308.01740).

## Recurring failure patterns

From issue trackers and papers, with what each means for our changes:

1. **An unconverged inner SCF poisons the outer step.** φ has several values for one N, and N runs away (CP-VASP #12, #6, VASPsol #18).
   - **Direct risk for our inexact Newton and adaptive criterion**, which feed loosely converged φ into the outer step on purpose.
   - The safeguards are the 1/λ weighting and the tight reconvergence before accepting. Both must be tested on tier 1–2 systems before the adaptive scheme becomes a default.
2. **φ(N) non-monotonic or discontinuous:** coarse k-points with small smearing; a cavity reaching into low-density metals; adsorbate levels crossing E_F; spin flips.
3. **Spill-out at low work function.** Adding electrons is much harder than removing them.
4. **Limit cycles in N**, especially when coupled to geometry steps. TPOT's workaround: equilibrate N with large steps first, then relax; for NEB, interpolate N between the endpoints.
5. **The counter-charge region saturates** after a big N step, and the run aborts.
6. **The state after an N change dominates cost.** Workarounds: extrapolating wavefunctions or density (CP-VASP: `NELMDL=2`, `IWAVPR=11`). A path-dependence bug in GPAW: grand energies 0.18 eV apart between constant-charge and constant-potential runs at the same N (gpaw-users, August 2023, fixed in !1462). This matches our finding that re-converging after the N jump is the per-attempt cost.

## Remedies reported in the literature, and what to take from them

| Remedy | Source | Relevance for SJM |
|---|---|---|
| Auxiliary-Hamiltonian variational GC-DFT, N inside the minimisation: "smooth exponential convergence", insensitive to vacuum size | Sundararaman, Goddard, Arias, JCP 146, 114104 (2017) | The robust end state. A plain grand-canonical SCF *struggled* there, so "N inside the SCF" needs care. |
| GC-SCF with modified Kerker G² → G² + q_κ², q_κ = 0.17 a₀⁻¹, q_K = 0.8 a₀⁻¹; a larger q_κ diverged; a residual 10⁻³ e oscillation remained | same | Concrete preconditioner numbers for idea 10 in `README.md` (inexact multisecant / GC-SCF). |
| Secant outer loop: μ within about 10 iterations, about 3× the cost of a fixed-charge run | same | That's our baseline; we're already better on easy metals. |
| Update N within each SCF step (density matrix as the only variable): about half the iterations of JDFTx | Q-Chem periodic GC-DFT (Ni, Rettig, Lee, arXiv:2508.15705) | Supports updating N inside the SCF. |
| Work-function constraint mixed toward the target with coefficient 0.3 (CP2K GCE) | Chai & Luber, JCTC 2024 (DOI unverified) | A damped, inner-loop N update. |
| CIP: control the bulk inner potential, which is continuous across gaps | Melander et al. 2024 | The right control variable for semiconductors. Already in old SJM (`pot_ref='CIP'`), but `inner_region` is broken. |
| Fixed μ for gapped systems | our reasoning: N(μ) comes directly from the occupations, and C_q ≪ C_dl means no sloshing | Metals are better served by controlling N, semiconductors by controlling μ. Switch on the C_q/C_dl ratio. |
| Safeguarded or bracketing root-finding in a transformed variable (e.g. asinh(N/N_s)), with the local slope from C_q at every attempt | our reasoning | Needed for tier 1 #5 and #7. |
| Constraints on the solute (GCE-cDFT, occupation matrices) | CIP paper; Fe–N–C papers | For tier 2 #9 and tier 3 #16–17; outside SJM's scope. |
| Avoid the iteration: canonical runs plus capacitor or electrostatic corrections | Gauthier et al., ChemPhysChem 2019; Domínguez-Flores & Melander, JCP 158, 144701 (2023) | A fallback for hopeless cases. |
| VASPsol: surface tension τ = 0 removes oscillation of the interfacial potential | Bhati et al. 2020 | Try `SurfaceInteraction` off on gapped systems. |

## Suggested first round

Use the existing harness, 1 core each, at most 2 at a time:
- Tier 1 #1 (Au Γ-only, 0.01 eV);
- #2 (Au at 2.0 V, close jellium);
- #3 (graphene at the Dirac point);
- #5 (Si(111)-H, with the target inside the gap and in each band).

Run each with master, slope estimate, inexact Newton and adaptive criterion. Plus one re-run of the easy Au case with a thick jellium, to check the 0.79 Å caveat.

## Sources

Main references (DOIs where the agents verified them):
- Kastlunger, Lindgren, Peterson, JPCC 122, 12771 (2018), 10.1021/acs.jpcc.8b02465
- Melander, Kuisma, Christensen, Honkala, JCP 150, 041706 (2019), 10.1063/1.5047829
- Melander, Wu, Weckman, Honkala, npj Comput Mater 10, 5 (2024), 10.1038/s41524-023-01184-4 (CIP-DFT)
- Moradi & Melander, Curr Opin Electrochem 49, 101615 (2025), 10.1016/j.coelec.2024.101615
- Sundararaman, Goddard, Arias, JCP 146, 114104 (2017), arXiv:1701.04490
- Bhati, Chen, Senftle, JPCC 124, 26625 (2020), 10.1021/acs.jpcc.0c07583
- Bonnet, Morishita, Sugino, Otani, PRL 109, 266101 (2012)
- Hagopian, Doublet, Filhol, Binninger, JCTC 18, 1883 (2022), arXiv:2201.04537
- Surendralal, Todorova, Finnis, Neugebauer, PRL 120, 246801 (2018)
- Andersson & Zhang, arXiv:2308.01740 (review)

Issue and forum reports:
- JDFTx #155, #288, #313, #432 (github.com/shankar1729/jdftx/issues)
- VASPsol #36, #49, #99 (henniggroup/VASPsol); VASPsol/VASPsol #18
- CP-VASP #6, #7, #11, #12, #23, #25, #41 (yuanyue-liu-group/CP-VASP)
- FCP-vasp-ase #3; TPOT #10 (comet-group/tpot)
- gpaw-users 2023-August/007120, 2024-April/007240, 2024-November/007385 (listserv.fysik.dtu.dk)

**Coverage gaps:** the QE users-list archive failed to load (certificate error); no CP2K or Matter Modeling StackExchange threads were found; VASP forum threads weren't opened; Nishihara & Otani PRB 96, 115429 (2017) wasn't accessible.
