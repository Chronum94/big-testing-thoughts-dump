> **Note:** written with Claude (Anthropic's AI assistant) under close human oversight. This is the precise specification of what was run, so the numbers in `README.md` can be reproduced and argued with. Where a description and the code disagree, the code (`gpaw/solvation/sjm.py` on branch `sjm-convergence`, `scripts/bench_setup.py`, `scripts/rootfind_sim.py`) is authoritative.

# Methods and terminology

## Notation

| symbol | meaning | units |
|---|---|---|
| N | excess electrons in the slab (`sj['excess_electrons']`); the jellium carries −N | e |
| φ | electrode potential: top-side work function, V_vac − E_F | V |
| φ* | target potential (`target_potential`) | V |
| f(N) = φ(N) − φ* | the residual of the outer problem | V |
| s = dφ/dN | slope (negative: adding electrons lowers φ) | V/e |
| C = −1/s | cell capacitance | e/V |
| C/A | areal capacitance, A = cell area in the xy-plane | µF/cm² |
| μ = −φ | electron chemical potential vs vacuum | eV |
| Ω = E − μN | grand potential (SJM's reported energy) | eV |
| k | outer attempt index (one attempt = one SCF at fixed N) | — |
| σ_k | noise level (standard deviation) of the measured φ_k | V |
| τ_X | SCF tolerance of criterion X (energy, density, eigenstates, work function) | — |

## A. The outer loop

### A0. master (old GPAW, `_equilibrate_potential`)

```
k = 0; N_0 given
loop while k ≤ max_iters (10):
    set jellium charge −N_k, nvalence = N_neutral + N_k
    _quick_reinitialization (keep wavefunctions; reset mixer, eigensolver, criteria)
    SCF at fixed N_k to the tolerances τ   →  φ_k
    if k > 0 and |φ_k − φ_{k−1}| > max_step (2 V) and |f_k| > |f_{k−1}|:
        N_k ← (N_{k−1} + N_k)/2 ; redo                 (step halving)
    if |f_k| < tol: return
    s_reg = ordinary least-squares slope of the last ≤ 4 points (N_i, φ_i)
    s ← 0.5·s_old + 0.5·s_reg          (if no s_old: s ← s_reg)
    if no slope yet: s = −1602.2 / (10·A[Å²])     (10 µF/cm²)
    N_{k+1} = N_k − f_k / s
```

### A1. Physical slope estimate (committed; `_estimate_slope`)

Computed once, after the first SCF, when no slope is known. All profiles are averaged over x and y on the fine grid, as functions of z (spacing h):
1. n̄(z) is the pseudo-electron density (summed over spins), ε(z) the dielectric, m(z) the jellium mask.
2. Induced-charge shape (the "rigid outward shift" model): g(z) = −dn̄/dz, set to zero where z < z_top (the highest atom), where g < 0, or inside the jellium. Normalise Σg = 1.
3. Counter-charge shape: j(z) = m(z)/Σm.
4. Charge per added electron: ρ(z) = j(z) − g(z), so Σρ = 0. Enclosed charge: q(z) = cumulative sum of ρ.
5. Double-layer capacitance (atomic units): **C_dl = A / (4π h |Σ_z q(z)/ε(z)|)**. This is 1D Gauss's law with D = 4πq/A and E = D/ε.
6. Quantum capacitance from Fermi–Dirac occupations, with θ_kn = f_kn / f_max, f_max = w_k·2/n_spins, and width σ_FD in Ha: **C_q = Σ_k f_max Σ_n θ(1−θ)/σ_FD** = DOS at E_F. Summed over the k-point/band communicator. Infinite if the smearing is zero.
7. **1/C = 1/C_dl + 1/C_q, s₀ = −Ha/C.** If anything is missing (no dielectric, no jellium, g ≡ 0), fall back to 10 µF/cm².

**Rule changes** in the slope update:
- A *guessed* slope (from this estimate or user-given) is **replaced** by the first regressed slope, not mixed with it. Later regressed slopes are still mixed 0.5/0.5.
- A regressed s ≥ 0 is discarded and the previous slope kept.

### A2. Inexact Newton with scaled tolerances (committed; `sj['inexact_newton'] = η`)

Loosening factor for attempt k:

```
λ_k = max(1, η·|f_{k−1}| / τ_wf)        with f_{−1} := tol   (so attempt 0 has λ = η·tol/τ_wf)
```

Applied to the SCF criteria, relative to the user's tolerances:

```
τ_E, τ_ρ, τ_wf  →  λ_k · τ          τ_eig  →  λ_k² · τ
```

The eigenstate criterion is a squared residual norm, hence the square.

**Acceptance rule ("tightening" or "reconvergence"):** if λ_k > 1 and |f_k| < tol, the attempt is **not** accepted. λ is set to 1, `scf.converged` is reset and the SCF continues **at the same N**, with no reinitialisation and `system_changes = []`, until the user's tolerances are met. Then |f| < tol is rechecked. An accepted point is therefore always tightly converged.

**Weighted regression:** `np.polyfit(N_i, φ_i, 1, w = 1/λ_i)`, which minimises Σ (φ_i − a − sN_i)²/λ_i², since a loose point's error scales with λ_i. With η unset, all λ = 1 and this is master's unweighted regression.

### A3. Adaptive potential criterion (harness only; `adaptive=dict(density, eigenstates, eta)`)

Non-final attempts (λ ≥ 2, forced also for attempt 0) use:
- τ_ρ = 10⁻² and τ_eig = 10⁻⁴ ("coarse"), or 10⁻³ / 10⁻⁶ ("medium");
- τ_E = 10³, which effectively switches the energy criterion off;
- in place of the work-function criterion, `AdaptiveWorkFunction`. With W_j the pair (bottom, top) of work functions at SCF iteration j:

```
spread_j = max over the two sides of ptp(W_{j−2}, W_{j−1}, W_j)      (ptp = max − min)
converged ⇔ spread_j < max(τ_wf, η · |W_j,top − φ*|)
```

It is not converged before 3 values exist. The final attempt is reconverged with the original tolerances and the original `WorkFunction` object (reset), exactly as in A2.

### A4. Grand-energy extrapolation (committed; `sj['energy_correction']`)

Using dΩ/dμ = −N and d²Ω/dμ² = −C (Legendre structure), with N and φ the values of the converged attempt:

```
Δμ = μ* − μ = φ − φ*
Ω(μ*) ≈ Ω(μ) − N Δμ − ½ C Δμ²,      C = −1/s  (quadratic term only if s < 0)
```

Always logged. Written into `results['energy']` and `results['free_energy']` only if enabled. Forces are not corrected.

## B. Mixer-history experiments (harness only)

MSR1 (`gpaw/old/mixer.py`) keeps, per evaluated input i:
- the input pseudo-density nᵢ (`nt_isG`) and its residual Rᵢ = n_out(nᵢ) − nᵢ (`R_isG`);
- the metric-weighted residual (`mR_isG`), where the metric is a reciprocal-space weight that damps long wavelengths (weight 20, σ 1.1);
- the PAW atomic density matrices and their residuals (`D_iasp`, `dD_iasp`, `mD_iasp`).

It builds secants against the newest pair:

```
s_i = n_i − n_last,      y_i = −(R_i − R_last)
```

It then takes a regularised multisecant quasi-Newton step from (n_last, R_last), with "good-Broydenness" and a backtracking/trust logic.

- **Reset (master):** `mixer.reset()` at every N update; all pairs are discarded.
- **Keep (B1):** no reset. The pairs from the old N stay and are mixed with pairs from the new N.
- **Re-anchor (B2):** at an N update, store the evaluated old pairs (lists truncated to `len(R_isG)`), then reset. When the mixer state is first n = [a₀, a₁], R = [R(a₀)] (i.e. the second mix call at the new N), prepend the old pairs translated by the anchor:

```
X_i' = X_i − X_L + X(a0)     for X in {n, R, mR, D, dD, mD},   i = 0 … L−1
```

  L is the last old pair, which coincides with the anchor after translation and is dropped. Every secant difference is preserved exactly; only the absolute position moves.

## C. 1D root-finders in the offline comparison (`scripts/rootfind_sim.py`)

### C0. Simulation protocol

- **Curves:**
  - Au and Pt: cubic spline through the constant-charge sweeps, linear extrapolation outside the sweep;
  - Cu: cubic polynomial least-squares fit through 14 tightly converged (N, φ) points from the Cu logs;
  - semiconductor: φ(N) = 5.0 − 0.05·asinh(N/10⁻⁶) − N/0.15, which is Boltzmann band tails around a ~1.4 eV gap in series with a 0.15 e/V double layer.
- **Start:** N₀ = 0. First slope s₀ = s_true(N₀)/0.65 for all methods, mimicking our estimate's 0.65× capacitance.
- **Measurement:** φ_meas = φ(N) + Gaussian noise with standard deviation σ.
  - **Tight mode:** σ = 1 mV always.
  - **Inexact mode:** σ = 0.01 V for the first evaluation, then σ_{k+1} = max(1 mV, 0.1·|φ_meas,k − φ*|).
- **Acceptance:** if |φ_meas − φ*| < 0.01 V and σ > 1 mV, re-measure at the same N with σ = 1 mV. This counts as one more evaluation, mirroring the reconvergence. Accept if that is within tol; otherwise continue, with the tight point added to the history.
- **Failure:** 20 evaluations without acceptance, or a non-finite step.
- **Statistics:** 300 seeds, with the same seed sequence for every method. Median and 90th percentile of evaluations over the successful runs; failure counts reported separately.
- **Plots:** median over seeds of log10 |φ_true − φ*| at each evaluation index. Runs that already converged are padded at log10(tol/10) = −3, so −3 means "done".

Each method maps the history (N_i, φ_i, σ_i), i ≤ k, to N_{k+1}. With no usable model yet, all methods take the Newton step N_{k+1} = N_k − f_k/s with the current slope s.

### C1. master

Unweighted least-squares slope over the last ≤ 4 points, mixed 0.5/0.5 with the previous slope, including the initial guess. A Newton step from the current point. (The simulation omits master's max_step halving, which never triggers on these curves.)

### C2. branch (committed A1 + A2 logic)

Weighted least-squares slope over the last ≤ 4 points with weights 1/σ_i, the `polyfit` convention. The first one replaces the guess; later ones are mixed 0.5/0.5; s ≥ 0 is ignored. Newton step.

### C3. Anchored linear multisecant

Differences relative to the **current** point k (the "anchor"), over the last ≤ 4 points:

```
δN_i = N_i − N_k,   δφ_i = φ_i − φ_k,   w_i = 1/(σ_i² + σ_k²)      (i < k)
s = Σ w_i δN_i δφ_i / Σ w_i δN_i²                                    (weighted LS line through the anchor)
N_{k+1} = N_k − f_k / s                                              (keep the previous s if s ≥ 0)
```

The weight uses σ_i² + σ_k² because each difference contains the noise of both points.

### C4. Anchored inverse-quadratic multisecant

Fit N as a function of φ, through the anchor, by weighted least squares over the last ≤ 4 points:

```
δN_i ≈ a δφ_i + b δφ_i²          minimise Σ w_i (δN_i − a δφ_i − b δφ_i²)²
Δ = φ* − φ_k
N_{k+1} = N_k + a Δ + b Δ²
```

It is used **only if all** of the following hold; otherwise it falls back to C3:
1. at least 3 points exist;
2. **the curvature is resolvable**: ptp{δφ_i} ∪ {0} > 5·max(σ of the last 3 points), i.e. the φ values span more than 5× the noise (see glossary);
3. a < 0, the right sign for dN/dφ;
4. |bΔ²| < 0.5·|aΔ|: the quadratic correction is smaller than half the linear step, so the model is used only where it is a correction, not an extrapolation of the parabola.

### C5. IQI, inverse quadratic interpolation (Brent's fast step)

Exact Lagrange interpolation of N(φ) through the last 3 points, evaluated at φ*:

```
N_{k+1} = Σ_{i} N_i Π_{j≠i} (φ* − φ_j)/(φ_i − φ_j)      (i, j over the last 3 points)
```

It needs 3 distinct φ values; otherwise it takes a 2-point secant (if its slope is negative) or the current slope.

### C6. Sidi's method, order k

Fit the polynomial p of degree m − 1 interpolating the last m = min(k+1, available) points, in the variable x = N − N_k. Its derivative at the current point is the linear coefficient: s = p′(N_k). Then take the Newton step N_{k+1} = N_k − f_k/s, keeping the previous s if s ≥ 0. k = 1 is the secant method; the order tends to 2 as k grows.

## D. Proposed but not implemented

**D1. Rigid-band DOS predictor.** Using the eigenvalues ε_kn, weights w_k, Fermi level μ₀ and occupation function f of the last SCF:

```
ΔN = Σ_kn w_k [f(ε_kn − μ0 − Δμ) − f(ε_kn − μ0)]     → Δμ(ΔN), monotone, 1D root
Δφ_pred(ΔN) = −ΔN/C_dl − Δμ(ΔN)
solve Δφ_pred(ΔN) = φ* − φ_k  for ΔN
```

C_dl comes from A1 at first, then is refitted from the data with the band-filling part removed:

```
1/C_dl,i = −(φ_i − φ_k + Δμ_DOS(N_i − N_k)) / (N_i − N_k)
```

The refit uses the anchored weighted fit of C3.

**D2. MSR1-like step control (parked).**
- Trust radius r in volts: the step is clipped to |Δφ_pred| ≤ r.
- Acceptance ratio ρ = (|f_k| − |f_{k+1}|)/(|f_k| − |f_pred|):
  - ρ > 0.75: r doubles;
  - ρ ≤ 0.25 or |f| grew: r halves and the step fraction ("greed") drops;
  - |f| grew by more than 2×: reject and backtrack.
- Steps with |f_{k+1}| < 3σ_{k+1} are not judged (noise-limited).
- Once f has changed sign, keep the bracket and replace any step that leaves it with an Illinois step.

## Glossary

- **Attempt (outer iteration):** one SCF at fixed N, producing one measurement φ_k.
- **SCF iteration (inner iteration):** one eigensolver + density update + mixing cycle within an attempt.
- **Residual:** the quantity driven to zero. In the outer problem f = φ − φ*; in the SCF, R = n_out − n_in.
- **Tolerance τ:** the threshold on a convergence criterion. GPAW's `work function` criterion is the ptp of the last 3 work-function values, not the error against the converged value.
- **Inexact Newton:** solve each outer step's inner problem only approximately, with accuracy tied to how far the outer problem is from converged. The tie is the **forcing term** (here η): the inner error is allowed to be about η·|f|.
- **Tightening / reconvergence:** continuing a loosely converged SCF at the same N with the original tolerances before accepting it.
- **Noise (σ):** the scatter of a measured φ around its fully converged value. For loose SCFs it is roughly proportional to the tolerance used.
- **Noise floor:** the smallest meaningful difference between two φ measurements, about 1 mV for tight SCFs here.
- **Secant / secant pair:** two points (N_i, φ_i) and (N_j, φ_j) whose difference quotient approximates the derivative.
- **Multisecant:** using several secant pairs at once, usually by least squares. In high dimensions, to learn a Jacobian; in 1D, to average noise (linear model) or to capture curvature (nonlinear model).
- **Anchored:** differences are taken relative to the current point and the fitted model passes exactly through it. Older points only shape the slope or curvature; the newest, best-converged point is not "averaged away". MSR1 anchors at the newest pair too.
- **Weighted least squares (WLS):** a fit that minimises Σ wᵢ·(residualᵢ)², with wᵢ ∝ 1/σᵢ², so noisy points count less.
- **Curvature is resolvable:** the measured points spread over a range of φ much larger than their noise (here more than 5× the largest σ of the last 3 points). Only then is a fitted quadratic coefficient a property of the curve rather than of the noise. When the points cluster near the root, their differences are mostly noise, and a fitted curvature would be random. That is exactly why IQI and Sidi fail in the noisy simulation.
- **Order of convergence:** p in |f_{k+1}| ≈ c|f_k|^p near the root. Secant 1.618; Muller and IQI 1.839; inverse polynomial or Sidi with k+1 points → 2; Newton 2.
- **Muller's method:** fit a quadratic φ(N) through 3 points and take its root nearest the current point.
- **Bracket:** two points with f of opposite sign. φ(N) is monotonic, so the root lies between them. **Regula falsi / Illinois:** secant steps kept inside the bracket; Illinois halves the stale endpoint's weight to avoid stagnation.
- **Trust region:** a maximum step size that grows after good steps and shrinks after bad ones. **Greed** (MSR1's term): the fraction of the model step actually taken.
- **Slope mixing:** averaging a newly regressed slope with the previous one (`sj['mixer']`, 0.5). It damps noise but drags the slope toward stale or guessed values.
- **Double-layer capacitance (C_dl):** the electrostatic capacitance between the electrode's excess charge and the jellium counter-charge, set by their separation and ε(z).
- **Quantum capacitance (C_q):** dN/dμ from filling states, the DOS at E_F (smeared). It acts in series with C_dl: 1/C = 1/C_dl + 1/C_q. Large for metals (negligible in series), small for semiconductors and graphene (dominant).
- **Rigid-band approximation:** adding or removing electrons only moves the Fermi level through fixed band energies; the bands themselves don't change.
- **Potential of zero charge (PZC):** φ at N = 0.
- **Spill-out:** at low work function or large N, electrons leave the slab into the vacuum, solvent or jellium well. φ(N) then flattens or loses its root.
- **Jellium:** a uniform smeared charge (here −N) filling a slab-shaped region, or the solvent cavity for `cavity_like`, to keep the cell neutral. It stands in for the electrolyte's counter-ions.
- **Dipole correction:** a sawtooth potential that cancels the cell's net dipole field, so the top vacuum region is field-free and V_vac is defined there.
- **Charge sloshing:** a slow, oscillating SCF mode in which charge moves back and forth over long distances, typical of large metallic cells. **Kerker preconditioning** damps it by down-weighting long-wavelength (small-G) density changes. The MSR1 "metric weight" plays a similar role.
- **Γ-only / k-point staircase:** with very few k-points (and small smearing) the occupied states form a few discrete levels. N(μ) and φ(N) become step-like rather than smooth.
- **Grand-canonical SCF (GC-SCF):** adjusting N inside the SCF loop toward a fixed μ or φ, instead of an outer loop of fixed-N SCFs.
- **CIP:** constant inner potential. The control variable is the electrostatic potential averaged inside the slab instead of the work function; it is continuous across band gaps.
- **FDT mode:** SJM's fluctuation–dissipation potentiostat for MD: one stochastic N update per MD step instead of a root-find.
- **Path dependence:** the outer loop's sequence of N values (and hence its cost) depends on small differences in measured φ. Single-run totals can differ by ±50% for methods with identical per-attempt cost.
