> **Note:** this study was carried out with Claude (Anthropic's AI assistant) under close human oversight.
> None of the conclusions here are final.

# Faster constant-potential convergence in GPAW's solvated jellium method (SJM)

Constant-potential SJM runs an outer root-find on the excess electron count N, and each step is a full SCF. Can we make that cheaper and more robust without hurting accuracy?

**Short answer so far:**
- **Yes, about 2.4× in wall time on three easy metal slabs.** Au, Pt and Cu(111) go from 322 s to 135 s in total, at the same final potential tolerance (0.01 V).
- **What got us there**, three things stacked:
  - a physical first guess for dφ/dN, and no longer mixing later slopes with that guess;
  - inexact Newton: early outer steps are converged only as far as they need to be;
  - an adaptive stopping rule for early SCFs: stop once the potential is good enough to aim the next step, ignoring the density and eigenstate residuals.
- **The grand-energy extrapolation Ω(μ*) ≈ Ω(μ) − NΔμ − ½CΔμ² checks out:** under 0.7 meV error at 0.1 V off target. So the potential tolerance can be loosened safely for energies (not forces).
- **Keeping the MSR1 mixer history across N updates is a trap.** Naive keeping is 1.4–4× worse. Re-anchoring the secants is harmless but gives nothing. The cost per outer step comes from the N perturbation itself, not from lost mixer memory.
- **Every benchmark cell is a single run** (one target, one starting N). Path dependence of the outer loop moves totals by up to ±50%. A robustness sweep is still to do.
- **The benchmark jellium is only 0.79 Å thick.** It is the GPAW test geometry: top at 14.5 Å in a 15 Å cell, bottom at top atom + 3 Å. All numbers here refer to that setup.
- **For the outer root-find, anchor the multisecant on the current point and weight it by measurement noise.** An offline comparison of seven 1D root-finders on φ(N) curves from our own data (section 12) found:
  - classic higher-order methods (inverse quadratic interpolation, Sidi) are fastest with clean data, but fail in up to 2/3 of runs once inexact-SCF noise is added;
  - the anchored, noise-weighted multisecant (linear, or inverse-quadratic when the curvature is resolvable) never failed;
  - it is about 1.7× faster than master on a synthetic semiconductor, and equal or better on the metals.
  Precise definitions of every method and term are in [`METHODS.md`](METHODS.md).
- **Hard systems haven't been tried yet.** A merged, ranked list from three research agents is in [`HARD_CASES.md`](HARD_CASES.md): Γ-only metals, spill-out, graphene, low-density metals, semiconductors, spin, explicit water, elongated cells.

**Distilled summary:** [`LEARNED.md`](LEARNED.md) collects everything learned about solvation/SJM by topic.

## Requirements / where things are

- **GPAW:** branch `sjm-convergence` in a local clone (off `master` at `ca5e191ba1`). Commit `0c6c1de0da`, "SJM: faster constant-potential equilibration", changes only `gpaw/solvation/sjm.py` and `gpaw/test/solvation/test_sjm.py`. It holds items 1–3 below. Not pushed or upstreamed; the code decision is pending.
- **The other experiments are harness-only** (`scripts/bench_setup.py`, monkey-patched into the calculator): the mixer-history variants and the adaptive criterion.
- **Baselines** run `master`'s `sjm.py`, imported as `sjm_master.py` (regenerate with `git show master:gpaw/solvation/sjm.py > sjm_master.py`).
- **Machine:** WSL2, Ryzen 7 5800X, GPAW 26.7.1b1 (devenv conda env). **One core per job, at most two jobs at a time** (user constraint). `OMP_NUM_THREADS=1`.
- **Test system** (`scripts/bench_setup.py`):
  - structure: fcc(111) 1×1×3 slab (Au, Pt or Cu), cell z = 15 Å, slab bottom at 6 Å;
  - calculation: FD mode, PBE, 2×2×1 k-points, Fermi–Dirac 0.1 eV;
  - solvent: `SJMPower12Potential(u0=0.18)` cavity, `LinearDielectric(78.36)`, surface tension 18.4 mN/m;
  - jellium top at 14.5 Å, potential tol 0.01 V unless stated;
  - default convergence: energy 5e-4 eV/electron, density 1e-4, eigenstates 4e-8, work function 1e-3 eV;
  - default mixer: MSR1, β = 0.05, 10 old densities, weight 20.

## 0. How old-GPAW SJM works (what we started from)

The cell holds three things:
1. a DFT slab carrying N excess electrons;
2. a jellium slab of charge −N in the solvent region (default: from top atom + 3 Å up to cell top − 1 Å);
3. an implicit dielectric whose cavity comes from a repulsive r⁻¹² potential.

The Poisson equation is ∇·(ε∇V) = −4πρ. An iterative dipole correction (`SJMDipoleCorrection`) makes the top of the cell field-free. The electrode potential is the top-side work function:

```
φ(N) = V_vac − E_F(N)                       (≈ 4.4 V  ↔  0 V_SHE)
s    = dφ/dN < 0,      C = −1/s  (double-layer capacitance of the cell)
```

Constant potential means solving f(N) = φ(N) − φ* = 0 (`_equilibrate_potential`):

```
repeat:
    rebuild jellium (charge −N), nvalence = N0 + N
    _quick_reinitialization: keep wavefunctions, rebuild density, RESET mixer/eigensolver/criteria
    full SCF at fixed N, tight tolerances
    if |φ − φ*| < tol: done
    s_reg = linregress(last ≤4 (N_i, φ_i));   s ← 0.5·s_old + 0.5·s_reg
    N ← N + (φ* − φ)/s                          (halve if a step overshoots by > max_step)
first slope guess: 10 µF/cm²   (s0 = −1602.2 / (10·A[Å²]) V/e)
```

The energy is the grand potential Ω = E − μN with μ = −φ (`grand_output`).

## 1. Code survey → ideas

We read `gpaw/solvation/sjm.py` (old GPAW), `gpaw/new/sjm.py`, `gpaw/extensions/jellium.py` and `gpaw/new/scf.py`.

**Findings:**
- `gpaw.new` SJM raises `NotImplementedError`.
- `FixedPotentialJelliumExtension` does the same outer loop inside SCF through `post_scf_convergence`, with `mixer.reset()`.
- `fix_fermi_level` support already exists in the occupations.

**Ranked idea list from the survey:**
1. Grand-canonical SCF: update N inside the SCF loop.
2. A physical slope guess.
3. An analytic grand-energy correction, so the tolerance can be looser.
4. Inexact Newton.
5. A bracketing/safeguarded root-finder.
6. Smooth jellium edges. A sharp mask moving with max(z) jumps by a grid spacing; estimate ~10 mV for q = 0.1 e, A = 50 Å², h = 0.2 Å.
7. Predicting N across MD steps.
8. A closed-form `SJMDipoleCorrection`: the slope is linear in `corrterm`, and the code gathers the full 3D grid several times per Poisson solve.

**Bugs found while reading (not fixed):**
- CIP with explicit `inner_region` is broken three ways:
  - `p.inner_region` should be `p.cip['inner_region']`;
  - `get_inner_potential` is called without `z`;
  - the `autoinner` default is truthy, so the inner_region/autoinner check always raises.
- In FDT mode:
  - `previous_*` are mutable default lists that grow without bound;
  - `results['excess_electrons']` holds the *next* N, not the one the energy belongs to;
  - the docstring says the capacitance is in µF, but the code uses e/V.

The user picked 2, 3 and 4 to implement and test on small systems in a new branch, using at most 1–2 cores.

## 2. Baseline: the slope mixer is the problem

Au, target 4.0 V, N₀ = 0: 6 attempts, 144 SCF iterations, 68 s.

The log shows the true capacitance is about 34 µF/cm², against the 10 µF/cm² guess. Mixing every new slope 50/50 with the previous one kept dragging the estimate back toward the guess. The steps were too short:

```
4.191 → 4.138 → 4.071 → 4.023 → 4.018 → 3.993 V
slopes used: −22.2, −14.2, −10.4, −8.6, −7.5   (true ≈ −6.4 V/e)
```

## 3. Physical slope estimate (committed)

### 3a. The first attempt failed: geometric capacitance from the top nucleus

The plan was C_geo = A / (4π ∫ dz/ε(z)), integrated from the top atom to the jellium centroid. It gave 7.3 µF/cm² against a true ~36. My first quantum-capacitance term also had a units bug: the occupation `_width` is stored in eV, not Ha, which made C_q 27× too small.

### 3b. Diagnosis: where the excess charge actually goes

Two constant-charge runs (N = 0 and 0.03; `scripts/induced*.py`, `data/induced*.txt`) showed:
- The induced charge centroid is at 12.04 Å, 1.33 Å above the top Au nucleus (10.71 Å). It sits exactly where ε(z) climbs from 1 to 78 (11.5–12.7 Å). So any single-plane estimate is hopelessly sensitive: moving the plane from 10.7 to 12.0 Å moves C_geo from 7 to 200 µF/cm².
- Electrostatics decomposition for ΔN = 0.03: Δφ = −0.185 V, of which ΔE_F = +0.070 V and ΔV_bulk = +0.074 V. So C_q is negligible (about 1100 µF/cm²), and the drop is electrostatic.
- A 1D Gauss model, ΔV = −∫ 4π q_enc/(Aε) dz, built from the **total** Δρ̃ (including PAW compensation charges) reproduces the Poisson solution to within 10%. With the pseudo-density alone it is 2× off.

### 3c. The model that went in (`_estimate_slope`)

Model the added electron as a rigid outward shift of the surface profile, then apply Gauss's law with ε(z):

```
g(z) ∝ max(0, −dn̄/dz)   for z > z_top, outside the jellium,   ∫g = 1
j(z) = jellium mask,                                          ∫j = 1
ρ(z) = j(z) − g(z),   q(z) = ∫₀^z ρ
C_dl = A / (4π |∫ q(z)/ε(z) dz|)                         [e/Ha]
C_q  = Σ_k w_k Σ_n θ(1−θ)/σ        (θ = f/f_max, Fermi–Dirac)
1/C  = 1/C_dl + 1/C_q,             s0 = −Ha/C            [V/e]
```

Two more rule changes:
- A guessed slope (from the estimate or the user) is **replaced**, not mixed, by the first regressed slope. Mixing happens only between regressed slopes.
- A regressed slope s ≥ 0 is ignored.

Estimated vs regressed capacitance: Au 22.9 vs ~34, Pt 16.4 vs ~26, Cu 32.9 vs ~40–60 µF/cm². That is consistently 0.6–0.7×, on the safe side.

| (SCF iterations / attempts) | Au 4.0 V | Pt 4.5 V | Cu 4.0 V |
|---|---|---|---|
| master | 144 / 6 | 190 / 5 | 199 / 6 |
| + slope estimate | 116 / 4 | 125 / 4 | 167 / 6 |

Most of the gain is the "replace, don't mix" rule.

## 4. Inexact Newton (committed as `sj['inexact_newton']`)

Loosen the SCF while φ is still far off:

```
λ_k = max(1, η·|Δφ_{k−1}| / τ_wf)          (Δφ_{−1} := tol)
τ_E, τ_ρ, τ_wf → λ_k τ ;    τ_eig → λ_k² τ
```

If a loose attempt lands within tol, it is **not accepted**. λ is set to 1 and the SCF continues at the same N, with no reinitialisation, until the original tolerances are met.

Loose points are noisy, so the slope regression is weighted (`np.polyfit`, w = 1/λ):

```
minimise Σ_i (φ_i − a − s N_i)² / λ_i²
```

**Results with η = 0.1:**
- **Before weighting:** Au 52, Pt 166, Cu 94 iterations.
- **After weighting:** Au 52, Pt 145, Cu 94. Pt stays worse than with the slope estimate alone (125): loose measurements still corrupt its slope.
- **η = 0.05:** Au 67, Pt 149, Cu 114.
- **η = 0.3 (before weighting):** Cu failed, oscillating between N values that should give nearly the same φ but gave potentials differing by 0.07 V.

**Bug found and fixed:** the first version reran the tightening step with the original `system_changes`, which restarted the SCF from scratch. It now passes `[]`. Checked: start at N*, tol 0.05 → loose attempt-0 φ = 3.994, then 7 more iterations to 4.003.

## 5. Grand-energy extrapolation (committed as `sj['energy_correction']`, always logged)

With dΩ/dμ = −N and d²Ω/dμ² = −C:

```
Δμ = μ* − μ = φ − φ*
Ω(μ*) ≈ Ω(μ) − N Δμ − ½ C Δμ²,      C = −1/s
```

**Validation** with constant-charge sweeps (`scripts/sweep.py`, `check_legendre.py`; `data/sweep_*.json`):
- dΩ/dφ = N to within 0.0026 e rms (Au) and 0.0023 e rms (Pt).
- Errors against a spline reference at φ*:

| φ − φ* (V) | raw (meV) | 1st order | 2nd order |
|---|---|---|---|
| Au −0.110 | −4.41 | +1.08 | −0.09 |
| Au −0.228 | −11.34 | +4.62 | +0.08 |
| Pt +0.155 | +10.86 | +1.56 | +0.28 |
| Pt −0.279 | −26.86 | +3.80 | −1.08 |

- **Live runs at tol = 0.1 V with the correction:** Au takes 46 iterations, and is 0.11 meV from the reference; with inexact Newton too, 36 iterations and −0.04 meV. Pt takes 98 iterations at −0.16 meV, and 79 iterations at −0.11 meV with inexact Newton.
- Forces are not corrected.

Unit test: `test_sjm.py` gains an `accelerate` variant (inexact Newton + energy correction) that checks the energy equals Ω − NΔμ. Both FD/old variants pass.

## 6. Side findings

- **Relaxation NaN (existing, unexplained).** A short constant-potential BFGS relaxation (top Au atom displaced 0.15 Å outward, `scripts/relax.py`) dies after 3–4 steps with a NaN in dH (`partition.py: assert not np.isnan(buf).any()`). It happens on **master** too.
- **`test_sjm_fdt.py` is stale.** It is skipped upstream ("too slow") and fails on master as well: it compares the input `fdt` dict (old keys `po_time`/`th_temp`) with the rebuilt dict (`dt`/`capacitance`/`resistance`/`temp_K`).

## 7. Convergence plots (`figs/convergence.png`)

The top row shows |φ − φ*| per outer attempt against cumulative SCF iterations; the lower rows show the per-iteration density residual and work-function change.
- `master` closes only about 30–50% of the remaining error per attempt.
- Every update to N throws the density residual back to about 10⁻² and the work-function change to 0.1–1 eV: each attempt is nearly a fresh SCF.
- Much of each attempt is spent jittering around the 10⁻⁴ density tolerance.

## 8. The mixer: MSR1 history experiments (harness only)

MSR1 (`gpaw/old/mixer.py:205`, Marks' multisecant rank-1) stores (input nᵢ, residual Rᵢ) pairs and builds secants against the newest pair:

```
s_i = n_i − n_last,     y_i = −(R_i − R_last)
```

`master` resets this history at every update to N.

### 8a. Naive keep: bad (`figs/keep_mixer.png`)

When N changes, the residual function itself shifts, R_{N+δN}(n) ≈ R_N(n) + Δ. The first secant that crosses the update attributes Δ to a density change, and the old absolute pairs pull the solution back toward the old fixed point.

| total SCF iterations: reset / keep | Au | Pt | Cu |
|---|---|---|---|
| master | 144 / 626 | 190 / 383 | 199 / 433 |
| slope estimate | 116 / 168 | 125 / 177 | 167 / 410 |
| inexact Newton | 52 / FAIL | 145 / FAIL | 94 / FAIL |

**What it looks like:** 30–50 iterations on a wrong energy plateau, with the residual growing exponentially. Au sits at −10.90 vs −10.27 eV, Cu at −14.5 vs −10.9. Inexact Newton's loose tolerances are "met" on the plateau, so its φ readings are garbage: N = 0.0095 gives 3.91 V, while N = 0.0079 gives 4.16 V.

### 8b. Re-anchored secant history: neutral

Set the old pairs aside. Once the first new pair (a₀, R(a₀)) exists, prepend them translated:

```
n_i' = n_i − n_L + a0,     R_i' = R_i − R_L + R(a0)
(likewise for the metric-weighted residuals and the PAW D, dD, mD lists)
```

This preserves every secant difference exactly. It was checked to trigger at every update to N.

| reset / keep / re-anchored | Au | Pt | Cu |
|---|---|---|---|
| master | 144 / 626 / 151 | 190 / 383 / 188 | 199 / 433 / 149 |
| slope estimate | 116 / 168 / 102 | 125 / 177 / 166 | 167 / 410 / 189 |
| inexact Newton | 52 / FAIL / 98 | 145 / FAIL / 72 | 94 / FAIL / 97 |

The mean cost per attempt (after the first, excluding 1-iteration tightening reruns) is **26.5 with reset vs 26.0 re-anchored**, over 36 attempts each. Totals move only because the outer trajectories differ.

**Conclusion:** the per-attempt cost is the cost of re-converging after the N perturbation. Lost Jacobian information isn't the issue.

## 9. Adaptive potential criterion for non-final attempts (harness only, best so far)

The existing logs answer whether the density needs to reach 10⁻⁴ to give a usable φ. The work function settles (spread < 0.03 eV) only once the density residual is about 10⁻³·⁵–10⁻⁴: φ and density converge together. But φ is often settled by iteration 6–18, while the attempt runs to iteration 20–52 waiting for the density residual's jittery tail and the 1 meV window.

**So non-final attempts:**
- coarse density 10⁻² and eigenstates 10⁻⁴, energy criterion off;
- replace the work-function criterion with:

```
converged when   max_side ptp(W_{j−2..j}) < max(τ_wf, η·|φ_j − φ*|)
```

- all non-final attempts are loose, including attempt 0;
- the accepted attempt is reconverged tight, as in section 4.

| variant | SCF iterations (Au + Pt + Cu) | wall |
|---|---|---|
| master | 533 | 322 s |
| + slope estimate | 408 | 251 s |
| + inexact Newton 0.1 | 291 | 182 s |
| **adaptive, coarse 10⁻²/10⁻⁴, η = 0.1** | **262** | **135 s** (Au 34, Pt 42, Cu 59) |
| adaptive, medium 10⁻³/10⁻⁶, η = 0.1 | 375 | 193 s |
| adaptive, coarse, η = 0.3 | Cu FAILS (oscillates 3.95–4.02 V, 10 attempts) | — |
| adaptive, medium, η = 0.3 | 345 | 177 s |

- Coarse with η = 0.1 is the first variant that properly fixes Pt: 73 iterations.
- Loose iterations are cheaper in wall time: about 0.52 s vs about 0.6 s per iteration. The cause is untested; probably less eigensolver work.
- The final tight reconvergence from a coarse state is now the biggest single cost (3–34 iterations).

## 10. Plane-wave mode

SJM runs only on real-space grids (FD, and LCAO, whose Hamiltonian also lives on a grid).
- **Old GPAW:** `create_hamiltonian` raises `NotImplementedError` for PW (`sjm.py:1185`), and so does the parent solvation calculator.
- **New GPAW:** `gpaw.new` SJM raises `NotImplementedError` in every mode. Some PW pieces exist (`ConjugateGradientPoissonSolver` with dielectric and dipole layer, an unused Fourier-space sawtooth solver, `FixedPotentialJelliumExtension`). Plain implicit solvation does work in `gpaw.new` PW mode (commit `35151cbc5d`).
- **Everything in this log is FD.**

## 11. Gapped electrodes, and using the DOS

**Why gapped electrodes are hard.** With E_F in the gap, the carriers are the smearing tails of the band edges:

```
N(φ) ∝ exp(±φ/σ),     φ(N) ≈ φ0 ∓ σ ln(N/N0),     C ≈ C_q ≈ N/σ
```

So the slope dφ/dN ≈ −σ/N changes by orders of magnitude.
- Secant steps undershoot from one side and overshoot into the wrong band from the other.
- A 4-point regression averages slopes that differ by about 100×.
- Deep in the gap the required N falls below SCF noise, so there is no usable root. (A thin slab has no real space-charge layer anyway, so this is a stress test, not physics.)

**What the DOS can and can't do as a first guess.**
- C_q = DOS(E_F) is already in the slope estimate (section 3).
- For metals the DOS alone is useless: Au has C_q ≈ 1100 µF/cm² against a total of about 34, so a DOS-only step would overshoot about 30×.
- For gapped and low-DOS electrodes the DOS is the right first guess. What matters there is the DOS *shape*, not the single value at E_F.

**Proposed rigid-band predictor** (not implemented):

```
ΔN = Σ_kn w_k [f(ε_kn − μ0 − Δμ) − f(ε_kn − μ0)]   → Δμ(ΔN)  (monotone 1D root)
Δφ_pred(ΔN) = −ΔN/C_dl − Δμ(ΔN)
```

Solving Δφ_pred(ΔN) = φ* − φ0 gives N for the target, or, the other way round, the voltage for a target N.
- It reduces to the current step for metals.
- It reproduces the exponential/band-edge behaviour, the √N of graphene and k-point staircases.
- It also says when a target needs a ΔN below noise level, i.e. when to stop.
- It misses band changes that depend on N (adsorbate level crossings, spin flips, polarons).

Combined with section 12, the plan is: keep the band filling exact (DOS), and learn only the near-linear electrostatic part (C_dl) from the measured points with the anchored weighted fit.

## 12. Offline comparison of 1D root-finders (`scripts/rootfind_sim.py`)

**Question:** in 1D, can several previous points help, as in multisecant mixers?
- With a *linear* model, extra points only average noise. A single secant already fixes the line.
- Extra history therefore pays off only through a nonlinear model: Muller, inverse quadratic interpolation (IQI), inverse polynomial interpolation, Sidi's method, with orders 1.84 to 2 vs the secant's 1.62.

**Test (no DFT).** Simulated outer loops on φ(N) curves from our own data:
- Au and Pt: constant-charge sweeps, splined;
- Cu: cubic fit through tightly converged log points;
- a synthetic semiconductor: φ0 − σ·asinh(N/N_s) − N/C_dl, with σ = 0.05, N_s = 1e-6, C_dl = 0.15 e/V;
- the same first slope for every method (0.65× the true capacitance);
- Gaussian noise: 1 mV for tight SCFs, or max(1 mV, 0.1·|f|) for inexact ones;
- 300 seeds.

Methods and noise model are specified exactly in `METHODS.md`.

Evaluations to |φ − φ*| < 0.01 V, as median / 90th percentile / failures out of 300 (the failure count is omitted when zero):

| method | Au | Pt | Cu | semi, conduction band | semi, band edge |
|---|---|---|---|---|---|
| **tight SCF** | | | | | |
| master (regress + mix) | 4/4 | 6/6 | 5/5 | 18/18 | 17/18 |
| branch (weighted, replace guess) | 3/3 | 5/5 | 6/6 | 16/16 | 17/17 |
| anchored linear | 3/3 | 5/5 | 5/6 | 10/10 | 11/11 |
| anchored inverse-quadratic | 3/3 | 5/5 | 4/4 | 9/9 | 9/9 |
| IQI (3-point) | 3/3 | 5/5 | 4/4 | 7/7 | 7/7 |
| Sidi k=2 | 3/3 | 5/5 | 4/4 | 9/9 | 8/10 |
| Sidi k=3 | 3/3 | 5/5 | 4/4 | 11/12 | 12/12 |
| **inexact SCF (η = 0.1)** | | | | | |
| master | 5/6 | 7/8 | 6.5/8 | 19/20 | 19/20 |
| branch | 5/6 | 7/8 | 7/8 | 18/20/6 | 18/19/1 |
| anchored linear | 5/6 | 6/8 | 7/8 | 11/12 | 11/12 |
| anchored inverse-quadratic | 6/7 | 7/8 | 7/8 | 11/13 | 10/12 |
| IQI | 6/9 | 8/13/5 | 7/12/2 | 11/17/20 | 10/16/17 |
| Sidi k=2 | 6/7 | 7/10 | 7/9 | 16/19/30 | 15/20/26 |
| Sidi k=3 | 6/7 | 7/11/5 | 7/10/5 | 19/21/190 | 18/21/141 |

Figures: `figs/rootfind_sim.png` (all methods, both noise levels) and `figs/rootfind_anchored_tight.png` (tight SCF; master, branch, the anchored methods).

- **Clean data:** curvature-aware methods save 1–2 evaluations on curved Cu and about half on the semiconductor. Au and Pt are nearly linear, so there is nothing to gain.
- **Noisy data:** exact interpolation through noisy, clustered points amplifies the noise. IQI and Sidi fail often: Sidi k=3 fails 190 of 300 semiconductor runs.
- **The anchored weighted methods never failed.** The anchoring and weighting matter more than the curvature term.
- **The branch's remaining 0.5 slope mixing costs it on Cu** (6 vs master's 5).
- **Every method shares the semiconductor's slow start** of about 6 evaluations, from a first slope that is orders of magnitude off. That is the job of the rigid-band predictor (section 11).
- **Caveats:** model curves (Pt starts at N = 0, outside its sweep, by linear extrapolation); unbiased Gaussian noise; evaluation counts, not cost.

## 13. Ideas discussed but not done

- **Inexact multisecant Newton / grand-canonical SCF.** Solve for x = (n, N) jointly, with residual (R(n,N), w·(φ−φ*)), updating N every few SCF iterations. Precondition the N component with C_est from section 3. Gate N updates until the density residual is below about 10⁻²·⁵, because early-SCF φ swings by 1–10 eV. Then optionally put N into the MSR1 vectors. This follows from section 8: the cost is the size of the N jump, so many small jumps should beat a few large ones.
- **Promote the adaptive criterion (section 9) into `sjm.py`**, replacing the scaled-tolerance `inexact_newton`.
- **Replace the regression + slope mixing** in the outer loop with the anchored weighted multisecant (section 12). Use the inverse-quadratic term only when the curvature is resolvable.
- **Rigid-band DOS predictor** (section 11), for the first step and as the band-filling part of the model.
- **MSR1-style step control** (a trust region with acceptance ratio, noise-aware bad-step limits, a bracket with an Illinois fallback). Discussed, then parked by choice; see `METHODS.md`.
- **A robustness sweep:** 3 targets × 3 starting N × 3 metals × methods, to get distributions instead of single runs.
- **Hard test systems:** see `HARD_CASES.md`. Key warnings from it:
  - An unconverged inner SCF poisoning the outer step is the most-reported real failure, and it is exactly the risk our inexact and adaptive schemes take.
  - For gapped electrodes, N(φ) ∝ exp(±φ/σ). φ(N) is then logarithmic, and secant steps or regression in N break down. Fix that with a local C_q slope at every attempt, bracketing in a transformed variable, or fixed-μ control (metals are better served by controlling N, semiconductors by controlling μ).
  - JDFTx 2017 found a plain grand-canonical SCF struggled. Their modified Kerker parameters (q_κ = 0.17, q_K = 0.8 a₀⁻¹) are a starting point for idea 1.

## Files

- `scripts/bench.py`: one benchmark run. `TAG TARGET N0 key=value...` with `element=`, `keep_mixer=`, `adaptive=dict(...)`; everything else goes to `sj`.
- `scripts/bench_setup.py`: the test system, plus the harness patches (keep/anchor mixer, `AdaptiveWorkFunction`).
- `scripts/run2.sh` with `jobs*.txt`: run a job list two at a time, one core each.
- `scripts/sweep.py`, `check_legendre.py`: Legendre validation.
- `scripts/induced*.py`, `estim.py`: induced-charge and capacitance-model diagnostics.
- `scripts/plot_convergence.py`, `plot_keep.py`: figures.
- `scripts/rootfind_sim.py`, `plot_anchored_tight.py`: offline root-finder comparison (runs in seconds, no DFT).
- `scripts/relax.py`, `anchor_check.py`: relaxation and anchoring checks.
- `data/results.jsonl` (all runs since the harness rewrite), `results_old.jsonl` (the first runs), `results_table.md`, `sweep_*.json`, `induced*.txt`.
- `logs/gpaw_txt_logs.tar.gz`: all GPAW text logs.
