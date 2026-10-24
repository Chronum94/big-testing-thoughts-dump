> **Note:** compiled with Claude (Anthropic's AI assistant) from six read-only research sub-agents, plus a few spot checks run by hand (marked **[verified]**). None of the conclusions here are final.

# GPAW feature scan: old vs new code, branches, and what to poke at

Snapshot of 2026-10-24.
- **Local tree:** `~/gpaw` on branch `sjm-convergence`, off local `master` `ca5e191ba1`, which is **162 commits behind `origin/master`**.
- **Upstream:** refs as last fetched (newest commit 2026-10-23).
- **Where file:line comes from:** citations without a ref point at the local tree. Where `origin/master` differs it is said explicitly (mainly solvation and the dipole layer).

## TL;DR

- **New GPAW is the default.** `GPAW()` silently falls back to old code only when it hits a legacy-only feature (see "Routing").
- **Where old code is still needed:**
  - linear-response TDDFT, XAS, ΔSCF, QMMM/CDFT, `external=`/`background_charge=`;
  - FD hybrids, GLLB-sc/TB09, LCAO hybrids (`ri`), LCAO meta-GGA;
  - the CG and etdm-lcao eigensolvers, the Broyden mixer and spin-mixer drivers;
  - LCAO ScaLAPACK/ELPA;
  - SJM's CIP and FDT modes.
- **Where new code is ahead:**
  - GPU (PW), single precision;
  - the PPCG eigensolver (default for FD/PW) and the MSR1 mixer default;
  - self-consistent SOC (LDA only), spin spirals, orbital magnetisation, Bader;
  - the D3 extension, spin-direction constraints, the fast Blöchl PW Poisson solver, `symmorphic=False`.
- **Silent behaviour changes to know about:**
  - `MixerDif`/`MixerSum` turn into plain Pulay on new code without any warning;
  - `augment_grids`, `order`, `stridebands`, `sl_auto` and `use_elpa` are accepted but ignored;
  - the default symmetry tolerance is 1e-5 on new code vs 1e-7.
- **Constant potential:** old-code SJM is being replaced. Upstream `PW-SJM-UI-and-logging` (102 commits, Kastlunger and Peterson) is a new FD+PW SJM with pluggable parts. It has the same outer-loop algorithm, so all our `sjm-convergence` ideas still apply. They port as subclasses (see "Solvation and SJM").
- **Your branches:**
  - keep `lcao-tiled-kernels` (rebase and open an MR) and the 3 unique commits of `scf-paw-speedups`;
  - salvage ideas from `mixing-experiment-2` and `improving-docs-1`;
  - delete the rest;
  - three branches exist locally only; push them to your fork first.

## Routing: which code runs (`gpaw/dft.py`)

`GPAW()` is `dft.py:GPAW`. There is no environment variable; the switch is `legacy_gpaw=None|True|False`, with `None` meaning "decide automatically". The automatic choice falls back to old code when:
- `Parameters()` raises `LegacyGPAWError`:
  - `interpolation != 117`, or `interpolation` set in PW mode;
  - an unknown eigensolver (`cg`, `etdm-lcao`);
  - xc given as something other than a string or dict;
  - `external`;
- `background_charge` or `external` is passed (`:1155-1160`);
- xc starts with GLLB, TB09, LCY or CAMY, or FD mode is used with EXX/PBE0/B3LYP (`:1224-1227`). The LCAO check at `:1221` is disabled by a dummy mode name (`'lcaooooooooooooo'`), so LCAO always goes to new code;
- reading a `.gpw` makes new code raise `LegacyGPAWError` (`:1197`).

**[verified]** `GPAW(mode='lcao', xc='PBE0')` raises an *empty* `LegacyGPAWError` and does not fall back. With `legacy_gpaw=True` it gives "LCAO mode does not support orbital-dependent XC". New LCAO has a stub `HybridXCFunctional` (`new/lcao/hybrids.py`, `calculate` returns 42.0) that the builder selects (`lcao/builder.py:68,83`), but in practice construction fails before it runs.

## Ground-state core (old | new)

| Feature | Old | New |
|---|---|---|
| FD / PW / LCAO | ✓ | ✓ (new LCAO builds its potential through an FD sub-builder, or PW with `experimental={'pw_pot_calc':1}`) |
| Real-space Poisson (FD, FFT/DST, moment correction, dipole, zero_vacuum) | ✓ | ✓ (reuses `gpaw/poisson.py`) |
| PW Poisson | dipole layer | ✓ dipole layer, charged, fast Blöchl variant; the CG solver only through solvation/SJM |
| Mixers | Pulay, MSR1, Broyden, FFT; separate/sum/diff/fullspin drivers | Pulay, MSR1, multi, no-mixing. **Default MSR1** (β 0.05, nmaxold 10, weight 20). **Broyden raises.** Spin drivers are silently dropped (`dft.py:440`) |
| Eigensolvers | davidson (FD/PW default), rmm-diis, cg, lcao, etdm-lcao, etdm-fdpw | **ppcg (FD/PW default)**, davidson, rmm-diis, etdm-fdpw, lcao, scissors; cg and etdm-lcao go to old |
| Occupations | ✓ shared `create_occ_calc` | ✓ |
| Symmetry and k-points | ✓ (tolerance 1e-7) | ✓ own wrapper, tolerance 1e-5, `symmorphic=False`, BandPath input |
| SCF criteria | shared `convergence_criteria.py` | same, through a `SimpleNamespace` shim context; custom criteria reaching other `ctx.ham/wfs/dens` attributes will break |
| Forces | all modes | all modes |
| Stress | PW only | PW only |
| `.gpw` | writes version 3 | writes version 7, reads 3–7; old reads new files only partially, untested for v7 |
| Parallelisation | domain/kpt/band, BLACS (`sl_*`, `sl_auto`, ELPA) | domain/kpt/band/spin; `sl_diagonalize`/`sl_lcao` only in PW/FD; **no LCAO ScaLAPACK/ELPA**; several old keywords silently ignored |
| GPU | ✗ | PW only, single or double precision (`parallel={'gpu':True}` or `GPAW_USE_GPUS`) |

The untracked `gpaw/new/notmixer.py` is a dead debug stub: nothing imports it, and `mix()` prints the shape and returns 1.0.

## XC and Hamiltonian extras (old | new)

| Feature | Old | New |
|---|---|---|
| LDA/GGA (built-in, libxc) | ✓ | ✓; GPU only for the built-in LDA and PBE |
| meta-GGA, PW/FD | ✓ | ✓, including MGGA stress |
| meta-GGA, LCAO | ✓ | ✗ (`add_to_ked` raises); the local `lcao-mgga` branch is an 8-line stub |
| r²SCAN | libxc name only (`MGGA_X_R2SCAN+MGGA_C_R2SCAN`) | same; `'r2SCAN'` gives "Unknown functional" |
| Hybrids, PW | ✓ | ✓ self-consistent, with forces; non-self-consistent HSE band structures |
| Hybrids, FD | ✓ (Γ only) | EXX/PBE0/B3LYP go to old; HSE06 goes to new, which builds no hybrid operator. Check this. |
| Hybrids, LCAO | ✓ (`backend='ri'`) | ✗: a stub, and it errors (see Routing) |
| vdW-DF, libvdwxc | ✓ | ✓, including vdW + MGGA |
| D3 / D4 | ✗ / ✗ | D3 as an extension / ✗ (upstream `cherrypick-dftd4`) |
| DFT+U | ✓ | ✓ (shared `hubbard.py`); U+J and linear-response U on upstream branches |
| SOC | post-processing ✓; self-consistent experimental | post-processing ✓; self-consistent ✓, noncollinear **LDA only** |
| Noncollinear / spin spirals | experimental / ✗ | ✓ LDA / ✓ PW (`qspiral`) |
| SIC (PZ) | **[verified]** crashes on an H atom in FD (`IndexError`) | **[verified]** crashes (`'SIC' object has no attribute 'kernel'`) |
| GLLB-sc, TB09 | ✓ | go to old |
| External potentials | full set (`external=`) | E-field and B-field as extension instances only; `create_external_potential` isn't wired in |
| Constraints | ✗ | `SpinDirectionConstraint` |

## Excited states and analysis (old | new | reads either `.gpw`)

- **Work with new code:** response (χ₀, RPA, df), G0W0 (from a file only), BSE/MPA, MOM, ETDM-fdpw, band structure, DOS/PDOS, Wannier and wannier90, Berry phase, Born charges, vibrations/NEB/IR, elph displacements.
- **Partial on new code:**
  - real-time TDDFT FD/LCAO: still defaults to legacy; the new path is serial with no `td_potential`;
  - LDOS helpers and elph supercell: run through compatibility shims, untested.
- **Old only:** linear-response TDDFT (Casida), XAS, ΔSCF, ETDM-LCAO (ported on upstream `new-directopt-lcao`, 74 commits).
- **New only:** orbital magnetisation, Bader.
- **Removed:** the tight-binding mode.

## Solvation and SJM

| Feature | Old FD | Old LCAO | Old PW | New FD | New LCAO | New PW |
|---|---|---|---|---|---|---|
| Implicit solvation | ✓ | ✓ | ✗ | ✓ | through the FD path, untested | ✓, but no density-dependent cavities |
| SJM | ✓ | ✓ | ✗ | upstream branch | untested | upstream branch |
| Jellium / background charge | ✓ | ✓ | – | ✓ (`FixedPotentialJelliumExtension`) | through FD | ✓ |
| Dipole correction | ✓ | ✓ | ✓ | ✓ on `origin/master` (`solvation/dipole_layer.py`) | through FD | ✓ |
| CIP / FDT | ✓ (CIP `inner_region` buggy) | ✓ | – | ✗ (rejected as legacy-only) | ✗ | ✗ |

**`origin/master` has already restructured solvation:**
- `gpaw.new.solvation` → `gpaw.solvation.extension`;
- the old pieces moved to `gpaw/old/solvation/`;
- `dipolelayer={'zero_vacuum': True}`;
- work functions come from `poisson_solver.vacuum_levels()`.

**Upstream `PW-SJM-UI-and-logging`** (`gpaw/extensions/electrochemistry/`):
- `SJM(potentiostat=, solvation=)`, where `Potentiostat(Extension)` has pluggable `ChargingScheme` (slab or cavity jellium with erf edges against PW Gibbs ringing), `PotentialOptimizer` (`DirectOptimizer`) and `PotentialProbe` (work function).
- **The algorithm is the old outer loop:**
  - N changes only after a fully converged SCF;
  - the slope is regressed over the last 4 attempts and mixed 0.2·old + 0.8·new;
  - 10 µF/cm² fallback;
  - mixer reset at every N update;
  - step halving.
- **Also new:**
  - logging of attempts, Ω and the apparent capacitance through a new `post_converged` hook;
  - a fresh `maxiter` budget per charge step;
  - unit tests for the optimizer logic.
- **It deletes** `gpaw/test/solvation/test_sjm.py` and `gpaw/new/sjm.py`.

**`sjm-convergence` against upstream.** `git merge-tree` gives:
- a trivial conflict at the `initialize` line in `gpaw/solvation/sjm.py` against `origin/master` **[verified]**, since upstream changed only 6 lines (renames);
- plus a modify/delete conflict on `test_sjm.py` against the PW-SJM branch.

**How our ideas port to their design:**

| idea | where it goes | effort |
|---|---|---|
| anchored multisecant root-finding | a `PotentialOptimizer` subclass (`record_attempt` already keeps the history) | easy |
| energy extrapolation | `write_results` / `post_converged` | easy |
| physical slope estimate | replace the 10 µF fallback; needs nt, ε, the mask and the DOS handed to the optimizer | small |
| inexact Newton / adaptive potential criterion | needs a handle on the SCF criteria | medium |
| N update inside the SCF | needs a per-iteration extension hook in `new/scf.py` | a core change; raise upstream |

## Branches

### Yours, newest first

All checks were read-only.

| branch | what | verdict |
|---|---|---|
| `sjm-convergence` | our SJM work (1 commit) | port the ideas to upstream PW-SJM; rebase is trivial |
| `lcao-tiled-kernels` | tiled BLAS-3 LCAO kernels (`c/lfc_tiled.cpp`); same commit locally, on origin and on fork; merges cleanly | **keep: rebase and open an MR** |
| `scf-paw-speedups` | 3 unique commits: batched D_asii symmetrisation (Cu32 0.69 → 0.12 s), vectorised LDA PAW correction (0.98 → 0.58 s), PW phase tables; top 4 duplicate the tiled kernels | **keep: separate small MR** |
| `mixing-experiment-2` | `gpaw/mixer.py` rewrite; conflicts (file moved upstream) | salvage ideas |
| `improving-docs-1` | copybutton, CSS, gettingstarted | salvage the small docs changes |
| `mixing-experiment`, `lcao-mgga`, `gpu-mgpuga`, `not-mixer`, `unwrap-lcao-h`, `gpu-test-new-eigensovler` | merged, or stubs | delete |

**Local-only** (not on your fork): `scf-paw-speedups`, `mixing-experiment-2`, `improving-docs-1`. Push them before cleaning up.

**Untracked files:**
- `O.PBE`: a generated setup; move it out of the repo.
- `PAWDISCUSSION.md`: your notes; keep, but outside the tree.
- `mixer.py.bak`, `notmixer.py`, `profile_output.lprof`: delete.

### Upstream (active since 2026-08-25), ⭐ = matches our interests

- ⭐ **SCF and mixing:**
  - `density-precond` / `density-precond-2`: right-preconditioned Pulay, about 33 lines;
  - `almost-always-update-potential`;
  - `eigensolver-mixer-default-suggestion`: 8 lines, "let the mixer loose";
  - `mix-ked`: mixing τ for MGGA.
- ⭐ **LCAO performance:** `dist-dens-lcao` (+41), `lcao-eighl` (+14), `P_aMi-experiment` (+19). These hit the same hotspots as `lcao-tiled-kernels`.
- ⭐ **GPU / ROCm:** `libxc-gpu-2`, `libxc-gpu-rocm-ci` (CI on MI210; RDNA4 untested), `lfc-multigpu-naive`.
- ⭐ **Electrochemistry:** `PW-SJM-UI-and-logging`; `cg-solvation` (merged).
- ⭐ **MGGA:** `mgga-paw-xc`, `mix-ked`.
- **Others:**
  - SOC rewrite stack: `new-soc`, `soc-parallel`, `soc-zora`;
  - `hubbard-u-j`, `linear-response-u`, `cached-exx`, `cherrypick-dftd4`;
  - response: `use_vector_fft_for_bse`, `symGW`, `elph_supercell_refactor`;
  - `new-directopt-lcao`;
  - performance: `vectorized-fftw`, `optimize-ppcg-further`, `optimize_xc_paw`;
  - plus a lot of CI/clang-tidy work.

**`origin/master` vs local master** (162 commits, fast-forward):
- the solvation restructure (above);
- `gpaw/new/xc/` split into a package;
- a new C++/GPU LFC (`c/new_lfc/`);
- MPI changes: mpi4py is no longer auto-initialised, `gpaw -P` removed;
- ROCm CI;
- the ClO setup updated to version 201.

## FAFO: one combined list (1–2 cores, ranked by insight per CPU-minute)

1. **Upstream PW-SJM against our Au(111) benchmark,** in a `git worktree` (never checked out in `~/gpaw`): attempts, SCF iterations, Ω for old-FD / new-FD / new-PW. This is the head-to-head baseline. About 10–20 minutes.
2. **`MultisecantOptimizer(PotentialOptimizer)` plus our physical slope** inside that framework, swapped in via `potentiostat={'optimizer': ...}`. No core edits.
3. **Mixer variants on the SJM slab:** copy `eigensolver-mixer-default-suggestion`'s MSR1/PPCG parameters (no rebase needed), then try `density-precond-2` in a worktree. Compare iterations per attempt, which is our measured bottleneck.
4. **Silent mixer change [cheap, issue-worthy]:** spin-polarised O₂ with `MixerDif(0.05, 5, 50)`, old vs new. New code silently uses plain Pulay; compare the magnetic moment and iterations. Seconds.
5. **Old vs new defaults:** H₂O in an FD box and an Al(111) PW slab, `legacy_gpaw=True/False`, then new with `eigensolver='davidson'`. Separates the eigensolver effect (PPCG) from the mixer effect (MSR1).
6. **PW Gibbs-ringing sensitivity in PW-SJM:** sweep jellium edge `width` = 0, 0.3, 0.6; log the dipole-layer net charge and φ drift.
7. **`scf-paw-speedups`:** rebase its 3 commits and re-time Cu32. Then extend the batched LDA PAW correction to GGA, the common production case.
8. **`mix-ked` with r²SCAN in PW** (LCAO MGGA is missing) on small Cu: does mixing τ remove the sloshing?
9. **ROCm build check** of `libxc-gpu-rocm-ci` on the RX 9070 XT (gfx1201). A build log alone answers whether it's feasible.
10. **Charged-interface analysis:** MOM vs ΔSCF-by-charge for a defect level; LDOS/PDOS and Bader on a charged SJM slab, to see where the counter-charge sits.

Issue-worthy findings: the empty `LegacyGPAWError` for LCAO hybrids; SIC crashing on both paths; spin-mixer drivers dropped silently; ignored parallel keywords; possibly FD + HSE06 running without a hybrid operator (unverified).
