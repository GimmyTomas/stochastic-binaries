# Numerical reproduction of Figures 2-7

One self-contained script per figure of the paper. All scripts accept
`--fast` (coarse smoke run) and `--out DIR` (default `output/`), write both a
PDF and the underlying data tables, and print quantitative gates.

| script | paper figure | what it does | default runtime |
|---|---|---|---|
| `fig2_adiabatic_eccentricity.py` | Fig. 2 (`fig:f-adiabatic`) | 1D FP for e, adiabatic tidal; relaxation to the thermal 2e | seconds |
| `fig3_whitenoise_ae.py` | Fig. 3 (`fig:f-white-noise`) | 2D FP in (ln a, e^2), white-noise tidal coefficients, absorbing at large a; heatmaps + marginals + f_ss overlay | ~2 min |
| `fig4_pointmass_ae.py` | Fig. 4 (`fig:f-point-mass`) | same for point-mass impulsive encounters at logLambda = 25; sub-thermal f_ss | ~2 min |
| `fig5_Td_vs_R.py` | Fig. 5 (`fig:Td(R)`) | T_d/T_d^* vs R/a for Gaussian perturbers (closed-form kernels); asserts the <9% tidal-limit claim | seconds |
| `fig6_fp_vs_mc.py` | Fig. 6 (`fig:fp-vs-mc`) | vectorized Monte-Carlo of impulsive kicks (1e6 binaries/case) vs the Fokker--Planck solution, uniform-sphere perturbers (Ramirez et al. 2022 setup) | ~15 min |
| `fig7_coefficient_convergence.py` | Fig. 7 (`fig:white-noise-numerics`) | direct orbit integration in an Ornstein--Uhlenbeck tidal field; measured/predicted coefficient ratios (App. D) | ~30 min (cached) |

Shared machinery in `common/`:

* `fp2d.py` — conservative finite-volume 2D Fokker--Planck solver (9-point
  stencil, backward Euler with a single sparse-LU factorization, per-edge
  reflecting/absorbing boundaries). Run the module directly for its
  self-tests (assembly equivalence, exact mass conservation, annihilation of
  the analytic stationary states).
* `fp1d.py` — the regular 1D solver for the adiabatic eccentricity problem.
* `coefficients.py` — every coefficient set used by the solvers, generated
  from the paper's (a, e) coefficients by the numerically-evaluated Ito
  transform; includes the validation gate that the general impulsive
  integrands with tidal-limit Q reproduce the closed-form white-noise
  coefficients. Run directly to execute the gate.
* `kick_covariance.py` — Q_r(r), Q_t(r) for uniform-sphere perturbers
  (Fourier route with an independent real-space quadrature gate).

Physics note for Fig. 6: the (R = 0.1 pc, a0 = 0.05 pc) case violates the
shot-noise condition (eqn:poisson-large), so the Fokker--Planck curve
deliberately disagrees with the Monte Carlo there (it over-ionizes); this is
discussed in the paper and is not a numerical artifact.
