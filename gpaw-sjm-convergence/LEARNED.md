> **Note:** distilled with Claude (Anthropic's AI assistant) under close human oversight from the work logged in `README.md`, `METHODS.md` and `HARD_CASES.md`, plus the GPAW feature scan (`../gpaw-feature-scan/`). None of the conclusions here are final. All numbers come from single runs on small test systems unless stated.

# What we learned about solvation and constant-potential SJM in GPAW

## 1. How SJM works, and what limits it

- **The model:** the slab carries N excess electrons. A jellium slab of charge −N sits in the implicit solvent, and the cavity comes from a repulsive r⁻¹² potential (dielectric ε 78). A dipole correction makes the top vacuum field-free.
- **The electrode potential** is the top-side work function, φ = V_vac − E_F (≈ 4.4 V ↔ 0 V_SHE). Energies are grand potentials, Ω = E − μN with μ = −φ.
- **Constant potential** is a 1D root-find, φ(N) = φ*. In old GPAW it's an outer secant loop: each step is a **full, tightly converged SCF at fixed N**, with the mixer, eigensolver and convergence criteria reset.
- **The cost structure:** about 3–6 outer steps × 20–40 SCF iterations each. Every step to a new N throws the density residual back to about 10⁻², so each outer step is nearly a fresh SCF.
- **The capacitance is electrostatic** in these metal cells: C ≈ 25–50 µF/cm², with C_q ≈ 1100 µF/cm² contributing only 2–5% of 1/C.
  - The excess charge sits about 1.3 Å above the top nucleus, exactly where ε(z) rises from 1 to 78. So any estimate based on a single "surface plane" is hopelessly sensitive: moving the plane from 10.7 to 12.0 Å changes C from 7 to 200 µF/cm².
  - A 1D Gauss model built from the **total** induced charge (PAW compensation charges included) reproduces the Poisson solution to within 10%. Using the pseudo-density alone is off by 2×.
- **φ(N) for metals is nearly linear,** because metallic screening makes the DOS irrelevant to the slope. Measured curvatures: Au 10, Pt 21, Cu 20 V/e². Curvature matters only when the target is far from the potential of zero charge (as for Cu: −0.16 e needed, about 1 V away).

## 2. What sped it up (FD; Au/Pt/Cu(111) 1×1×3 at tol 0.01 V; totals over all three)

| change | SCF iterations | wall time | status |
|---|---|---|---|
| master | 533 | 322 s | — |
| physical slope estimate, and **replacing rather than mixing** the guessed slope | 408 | 251 s | committed (`sjm-convergence`, `0c6c1de0da`) |
| + inexact Newton (loosen the SCF in proportion to \|Δφ\|; reconverge tightly before accepting) | 291 | 182 s | committed, `sj['inexact_newton']` |
| + adaptive potential criterion (coarse density/eigenstates; stop once the work-function spread < max(1 meV, 0.1·\|φ−φ*\|)) | 262 | **135 s (2.4× faster)** | harness only |

- **The 50/50 slope mixer was master's main problem.** The 10 µF/cm² guess is 3–6× off, and mixing every new slope with it drags it back toward the guess forever.
- **The physical slope estimate is mediocre:** a rigid-shift induced charge plus Gauss's law with ε(z) gives 0.6–0.7× the true capacitance. That's on the safe side, but it's the "replace, don't mix" rule that does most of the work.
- **Loose φ readings are noisy.** Weighting the slope regression by 1/λ (λ = the loosening factor) was necessary, and η = 0.3 is too loose (Cu oscillated and failed). η = 0.1 is the working value.
- **φ and the density converge together:** φ settles to 0.03 eV only once the density residual is about 10⁻³·⁵–10⁻⁴. The adaptive criterion gains by stopping as soon as φ is good enough for the next step (often SCF iteration 6–18), instead of waiting out the jittery 10⁻⁴ density tail (iterations 20–52).

## 3. Grand-energy extrapolation (committed, `sj['energy_correction']`, always logged)

- **The formula:** Ω(μ*) ≈ Ω(μ) − N Δμ − ½ C Δμ², with Δμ = φ − φ*.
- **Validated against constant-charge sweeps:**
  - dΩ/dφ = N to within 0.003 e;
  - at 0.1 V off target, the error drops from 2–10 meV to under 0.7 meV.
- **In practice:** tol = 0.1 V with the correction gives energies within 0.1–0.2 meV in a third of master's iterations.
- **Forces are not corrected.**

## 4. Mixer history across N updates (MSR1)

- **Naive keeping is bad:** 1.4–4× more iterations, and inexact Newton fails.
  - The old secant pairs belong to the old residual function: R_{N+δN}(n) ≈ R_N(n) + Δ. The crossing secant blames Δ on a density change.
  - The SCF then sits for 30–50 iterations on a wrong plateau (Au −10.90 vs −10.27 eV) while the residual grows.
- **Re-anchored secants are neutral:** translate the old pairs onto the first new pair, keeping every difference. This gives 26.0 vs 26.5 iterations per attempt.
- **Conclusion:** the per-step cost is re-converging after the N perturbation itself, not lost Jacobian information. The lever is fewer or smaller N jumps, not mixer memory.

## 5. Outer root-finder (offline simulation on φ(N) curves from our own data)

- **In 1D, extra history only helps through a nonlinear model.** A single secant already fixes a line; more points with a linear model just average noise.
- **IQI and Sidi are fastest on clean data.** On curved Cu and a synthetic semiconductor they take 7–9 evaluations against 16–18. But under inexact-SCF noise they **fail in up to 2/3 of runs**: exact interpolation through noisy, clustered points.
- **The anchored, noise-weighted multisecant never failed** (a line through the current point, inverse-quadratic only when the curvature is resolvable above 5× the noise).
  - It's about 1.7× faster than master on the semiconductor curve, and equal or better on the metals.
  - **This is the recommended replacement for the regression + mixing.**
- **The branch's remaining 0.5/0.5 mixing** after the first regression costs it an evaluation on Cu. Anchoring removes the need for it.

## 6. Metals vs semiconductors: which variable to control

- **The slope is** dφ/dN = −(1/C_dl + 1/C_q).
  - **Metals:** C_q ≫ C_dl, so the slope is moderate and nearly constant. **Control N.** Fixed-μ control sloshes, because N(μ) is steep.
  - **Gapped electrodes:** C_q ≈ N/σ, so N(φ) ∝ exp(±φ/σ) and φ(N) is logarithmic, nearly vertical near N = 0. **Control μ** (or φ). Secant steps in N overshoot into the wrong band edge, and regression averages slopes that differ by 100×.
- **One formulation covers both:** root-find in u = Δμ_DOS(N), the rigid-band band-filling shift computed from the eigenvalues:
  - φ ≈ φ₀ − u − N/C_dl;
  - learn only C_dl (near-linear) from the data, with the anchored multisecant.
- **The DOS alone is useless as a slope for metals** (a step 30× too large) but is the right model for gapped and low-DOS electrodes (semiconductors, graphene √N, Γ-only staircases).
- **Deep in the gap,** the required N falls below SCF noise: no usable root. The loop should detect this and stop. Physically, a thin slab has no space-charge region, so "potential in the gap" is a model artefact.
- **CIP (constant inner potential)** controls the bulk electrostatic potential instead. Its slope is pure 1/C_dl, near-linear even for semiconductors, but it fixes band positions rather than E_F (a modelling choice). In old SJM, `inner_region` is broken in three places; only `autoinner` works.
- **Charged-defect occupations** at electrode surfaces are a natural application. Transition levels appear as steps in N(μ), and the scan exposes semilocal delocalisation error (smeared plateaus). Rigid bands fail at the defect level itself, so each scan point has to be self-consistent.

## 7. GPAW code state (as of 2026-10-24)

- **Modes:**
  - Old SJM: FD and LCAO only; PW raises.
  - `gpaw/new/sjm.py`: raises `NotImplementedError` in every mode.
  - Plain implicit solvation in new PW mode works, but without density-dependent cavities.
- **`origin/master` restructured solvation:**
  - `gpaw.new.solvation` → `gpaw.solvation.extension`;
  - old code moved to `gpaw/old/solvation/`;
  - `dipolelayer={'zero_vacuum': True}`;
  - work functions come from `poisson_solver.vacuum_levels()`.
- **Upstream `PW-SJM-UI-and-logging`** (102 commits, Kastlunger and Peterson): a new FD+PW SJM in `gpaw/extensions/electrochemistry/`.
  - Pluggable `ChargingScheme` (slab or cavity jellium with **erf edges against PW Gibbs ringing**), `PotentialOptimizer` and `PotentialProbe`.
  - Logging of each attempt, Ω and the capacitance through a `post_converged` hook.
  - **Same outer loop as old SJM:** full SCF per N, regression over 4 points with 0.2/0.8 mixing, 10 µF/cm² fallback, mixer reset, step halving.
  - CIP and FDT are rejected as legacy-only; LCAO is untested.
- **Porting our ideas onto upstream:**

| idea | where | effort |
|---|---|---|
| anchored multisecant | `PotentialOptimizer` subclass | easy |
| energy extrapolation | `post_converged` | easy |
| physical slope | needs nt, ε, mask and DOS passed to the optimizer | small |
| inexact Newton / adaptive criterion | needs a handle on the SCF criteria | medium |
| N update inside the SCF | needs a per-iteration extension hook in `new/scf.py` | raise upstream first |

- **Our branch against upstream:** a trivial conflict at `initialize` in `gpaw/solvation/sjm.py` (upstream only renamed `OldSolvationGPAW` → `SolvationGPAW`). Upstream PW-SJM also deletes `test_sjm.py`.

## 8. Bugs and oddities found (not fixed)

- **CIP with explicit `inner_region` is broken:** `p.inner_region` should be `p.cip['inner_region']`; `get_inner_potential` is called without `z`; the truthy default `autoinner` makes the inner_region/autoinner check always raise; and the CIP checks run before `p.update(sj_changes)`.
- **FDT mode:**
  - `previous_*` are mutable default lists that grow without bound;
  - `results['excess_electrons']` is the *next* N, not the one the energy belongs to;
  - the docstring says µF where the code uses e/V.
- **`test_sjm_fdt.py`** (skipped upstream) fails on master as well: it compares old fdt keys (`po_time`/`th_temp`) against the rebuilt dict.
- **A short constant-potential BFGS relaxation hits a NaN** in dH after 3–4 steps, on master too.
  - Possibly related: a gpaw-users report (2024-11) of SJM NEB NaNs, traced to cavity objects shared between images and a too-thin bottom vacuum. Unverified.
- **`SJMDipoleCorrection`:**
  - its secant loop on `corrterm` is unnecessary, since the slope is linear in it (a closed-form solve would do);
  - it gathers the full 3D fine grid to every rank several times per Poisson solve;
  - `sjm_sawtooth` sizes its array with `int(L/h)`.
- **The sharp jellium mask** moves by a grid spacing as max(z) crosses grid planes, which is about 10 mV of noise in φ(R) for typical charges. Upstream's erf edges address the PW version of this.
- **Our benchmark jellium is only 0.79 Å thick** (GPAW test geometry: top 14.5 Å in a 15 Å cell, bottom at top atom + 3 Å). All capacitance and speed numbers refer to that setup.

## 9. What we expect to be hard (from literature and issue trackers; see `HARD_CASES.md`)

- **The most-reported real failure:** an unconverged inner SCF poisoning the outer step (CP-VASP, VASPsol). That's exactly the risk inexact Newton and the adaptive criterion take. The 1/λ weighting and tight-before-accept are the safeguards, and they're untested on hard systems.
- **Other hard regimes:**
  - spill-out at low work function (adding electrons is much harder than removing them);
  - non-monotonic φ(N) (coarse k-points with small smearing; a cavity inside low-density metals like Mg or Na; adsorbate levels crossing E_F; spin flips);
  - limit cycles when coupled to relaxations;
  - saturation of the counter-charge region;
  - the FD Poisson solver in elongated cells.
- **JDFTx 2017:** a plain grand-canonical SCF struggled. Their modified Kerker (q_κ = 0.17, q_K = 0.8 a₀⁻¹) is the starting point for an in-SCF N update.
- **Cheapest hard tests to run first:** Au Γ-only with 0.01 eV smearing; Au at 2 V with a close jellium; graphene at the Dirac point; Si(111)-H inside the gap; a thick-jellium rerun of the easy Au case.

## 10. Open directions, in order of expected payoff

1. Port to upstream PW-SJM: an anchored-multisecant `PotentialOptimizer` plus the physical slope estimate (no core changes).
2. The adaptive potential criterion as the implementation of inexact Newton (needs SCF-criteria access).
3. The rigid-band DOS variable u, for gapped and low-DOS electrodes.
4. An in-SCF N update with Kerker-type preconditioning, gated until the density residual is below about 10⁻²·⁵. It needs a per-iteration hook upstream.
5. A robustness sweep (several targets and starting N per metal) and the tier-1 hard cases, before changing any default.
