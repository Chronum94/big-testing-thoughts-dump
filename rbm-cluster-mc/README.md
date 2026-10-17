> **Note:** this study was carried out with Claude (Anthropic's AI assistant) under close human oversight.
> None of the conclusions here are final.

# Can Boltzmann machines recommend cluster Monte Carlo updates?

A small NumPy codebase for trying this out on 1/2/3-D Ising and q-state Potts models with arbitrary bonds
(uniform, ±J, Gaussian, dilute, staggered, fully frustrated). The pipeline generates a dataset with single-flip
Metropolis and trains a restricted Boltzmann machine (RBM) on it. The RBM's block-Gibbs samples are then used as
Metropolis–Hastings proposals, so results stay exact however good or bad the RBM is. It also implements L. Wang's
higher-order Boltzmann machines (PRE 96, 051301(R) (2017)), whose hidden units couple to *products* of spins and
which reproduce Swendsen–Wang (SW), Niedermayer and Kandel–Ben-Av–Domany (KBD) cluster algorithms. On top of those
it has a "feature-learning" variant that tries to learn which features to use.

It is meant for teaching and methods research. Readability came first, then correctness checked against exact
enumeration; speed came last.

## Requirements and quick start

Python ≥ 3.10 and NumPy (tested with 2.4); pytest for the checks. Nothing else.

```bash
python -m pytest -q test_checks.py                       # 20 exactness checks, ~75 s
python run.py --d 2 --L 8 --T 2.4                        # dataset -> RBM -> single-flip vs RBM-assisted chain
python run.py --d 2 --L 6 --q 3 --T 1.0                  # 3-state Potts
python run.py --d 2 --L 6 --bonds pm --T 1.0 --radius 2 --block 2   # ±J glass, sparse RBM, local block moves
```

| file | contents |
|---|---|
| `model.py` | lattices and bonds, energy, single-flip Metropolis, Wolff, dataset generation |
| `rbm.py` | categorical-visible RBM (PCD training, sparse mask), exact RBM-MH proposals (full/block/k-step), exact bond-RBM for any Ising model, FK-gated RBM cluster moves |
| `bm3.py` | Wang's higher-order BMs: link BM (SW/Niedermayer family, any bond signs), plaquette-Ising BM, frustrated-plaquette (KBD) BM, `FeatureBM` (a dictionary of candidate features, with free or rejection-free parameters) |
| `run.py` | end-to-end experiment and integrated autocorrelation time (Sokal windowing) |
| `test_checks.py` | exactness checks against enumeration and transfer matrices, each paired with a negative control showing the test can fail |
| `explore/` | smoke-test experiments (`*.py`) and raw outputs (`*.out`). **`explore/NOTES.md` is the full lab log; start with the synthesis at its top.** Brainstormed directions are in `IDEAS*.md` |

## Method in one line

A trained machine with marginal p_θ(s) proposes s → s′ through any kernel that is reversible with respect to
p_θ, for example s → h → s′ block Gibbs or a cluster flip that conserves E(s, h). The proposal is then accepted with

```
A(s → s') = min(1, exp(d(s') − d(s))),   d(s) = −βE(s) − log p_θ(s)
```

so only log p_θ (the RBM free energy, or a product of softplus terms) is needed, never a sum over hidden units.
If p_θ ∝ π, every move is accepted.

## Salient results

All runs are small smoke tests on a laptop CPU (N ≤ 1024 spins), with 1–8 seeds per point. τ is the integrated
autocorrelation time of a symmetry-invariant observable. Section numbers refer to `explore/NOTES.md`.

**1. A pairwise RBM only ever makes local moves (§1, §5–§7).** In an RBM the hidden units couple to linear
combinations of spins, so given h the spins are independent. Even the *exact* bond-RBM, which accepts every move,
has z = 2.03 ± 0.14 at the 2D Ising T_c (L = 8–24), the same as single-flip dynamics. Trained dense RBMs learn a
rank-1 magnetisation (Curie–Weiss) mode; sparse ones learn local magnetisation blobs.

**2. Every apparent RBM speed-up disappeared against the right control (§8b, §10, §11).** Example: Potts q=3,
L=6, T=1.0, 8 chains:

| extra move per sweep | τ_E | τ_m |
|---|---|---|
| none (single flip) | 12.1 ± 0.8 | 15.2 ± 1.1 |
| full-system RBM | 8.9 ± 0.5 | 11.9 ± 0.8 |
| **one extra Metropolis sweep (work-matched null)** | **4.5 ± 0.1** | **5.4 ± 0.2** |

Better training losses raise acceptance (0.12 → 0.19) without improving τ.

**3. Cluster moves come from coupling to spin *products*, not from training (§9, §12, §13).** Wang's link BM and
our FK-gated RBM both reach SW scaling only at the full Fortuin–Kasteleyn point; anywhere else τ grows like L².
Reproduction of Wang's Fig. 4 (plaquette Ising, K/J = 0.2, T = 2.4955, one run per point):

| L | local τ_E | BM-cluster τ_E | Binder ratio (→ 1.1679) |
|---|---|---|---|
| 8 | 7.9 | 3.6 | 1.162 |
| 16 | 21 | 4.9 | 1.177 |
| 32 | 127 (run too short) | 6.0 | 1.150 |

At the paper's own parameters the conditional model in this construction has 2.6% antiferromagnetic links and 8.6%
frustrated plaquettes. The paper doesn't discuss this, though the update remains exact.

**4. Frustration: Metropolis + KBD is ~15× cheaper on the fully frustrated Ising model (§14).** Wang's Appendix B
has no numerics. On its own, KBD is nearly microcanonical at low T and fails to equilibrate. Paired with Metropolis,
it decorrelates the critical modes almost every step (L = 16; energies checked against an exact transfer matrix):

| T | sampler | τ of S(0,0) | max τ × ms/step |
|---|---|---|---|
| 0.5 | Metropolis | 48 | 75 |
| 0.5 | KBD only | 311 (not equilibrated) | 248 |
| 0.5 | **Metropolis + KBD** | **1.1** | **5.4** |
| 0.3 | Metropolis | 62 | 97 |
| 0.3 | **Metropolis + KBD** | **1.0** | **6.3** |

On the ±J glass, the SW + KBD hybrid loses to Metropolis.

**5. Feature learning does not find KBD; a structural scan does (§15).** Optimising the weights of a dictionary of
candidate features for mixing failed in both free and rejection-free parametrisations. The reason is that the
landscape is flat except at one vertex:

| how each bond's coupling is split (FFIM, L=8, T=0.5) | ESJD | frozen bonds percolate |
|---|---|---|
| Metropolis only | 1.54 ± 0.02 | – |
| any share on single bonds (SW … 90 % plaquette) | 1.26–1.29 | 98–100 % |
| split across both checkerboard sublattices | 1.43 | 84 % |
| **all on one sublattice = KBD** | **4.38 ± 0.25** | **1 %** |

In this family the structure choice is discrete, and a smooth optimiser has nothing to follow.

**6. In glasses the usual training diagnostic is blind (§6b).** On a 4×4 ±J glass at T = 0.4, an RBM trained on
single-flip data has KL(π‖p_RBM) = 21–39 nats while its held-out std of d is ≤ 0.5. PCD also fails on its own,
even when trained on exact samples.

## Methodological lessons

- **Compare against work-matched nulls,** not just against single-flip.
- **Use symmetry-invariant observables.** A percolating cluster flip makes single-spin τ look like 0.5 while the
  physics is frozen.
- **Stuck or near-microcanonical chains make Sokal's τ look *small*.** Check means against an exact reference.
- **One-step π → π tests can have little power;** pick parameters where an error is visible and keep a negative
  control.
- **Never validate on held-out data drawn from the same chain** you trained on.

## Caveats

All results are small-scale smoke tests: few seeds, small L, and usually a single temperature. Several conclusions
apply only to the feature dictionaries and models tried. All the code is plain NumPy with a pure-Python union-find
and Metropolis loop, so cost comparisons (ms/step) carry a vectorisation confound, which is flagged where it matters.

## References

- L. Huang, L. Wang, *Accelerate Monte Carlo simulations with restricted Boltzmann machines*, PRB 95, 035105 (2017), arXiv:1610.02746
- L. Wang, *Exploring cluster Monte Carlo updates with Boltzmann machines*, PRE 96, 051301(R) (2017), arXiv:1702.08586
- D. Kandel, R. Ben-Av, E. Domany, PRL 65, 941 (1990); PRB 45, 4700 (1992) (KBD)
- F. Niedermayer, PRL 61, 2026 (1988); N. Kawashima, J. E. Gubernatis, PRE 51, 1547 (1995), arXiv:cond-mat/9411110
- S. Pilati, P. Pieri, PRE 101, 063308 (2020), arXiv:2003.09765 (sparse RBMs)
- A. Decelle, C. Furtlehner, Chin. Phys. B 30 (2021), arXiv:2011.11307 (RBM mean-field theory)
- S. Pilati, E. M. Inack, P. Pieri, PRE 100, 043301 (2019), arXiv:1907.00907
- Y. Nomura, A. S. Darmawan, Y. Yamaji, M. Imada, arXiv:1709.06475
- T. Kassis, V. Agarwal, Y. He, D. Patel, A. M. Brueckner, *Scientific Agent Skills: A Library of Procedural
  Knowledge for Research Agents*, arXiv:2609.00065 (2026) (the scientific-critical-thinking skill used during this study)
