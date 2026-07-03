# stochastic-binaries

Companion code for the paper

> **Evolution of Binaries Under Stochastic Perturbations**
> Andrea Caputo, Giovanni Maria Tomaselli, Chris Hamilton
> (arXiv: TODO)

The paper develops a unified Fokker--Planck framework for Keplerian binaries
subject to stochastic perturbations -- adiabatic tidal fields, white-noise
tidal fields, and impulsive encounters with perturbers of arbitrary density
profile -- and provides ready-to-evaluate drift and diffusion coefficients
for all six orbital elements. This repository lets you **verify every
analytical result and reproduce every numerical figure** of the paper.

## Contents

| directory | contents |
|---|---|
| [`symbolic/`](symbolic/) | SymPy scripts that re-derive **all** drift/diffusion coefficients from first principles and check them symbolically against the paper, in every parametrization: (a, e, Omega, i, omega, M0), (E, J), and the body frame (e-hat, q-hat, J-hat). Includes the general-perturber impulsive results as exact functionals of Q_r(r), Q_t(r), the point-mass closed forms with the exact Coulomb logarithm, the traceless (gravitational-wave) variants, steady states, Table I form factors, and the Euler-angle impulsive sector that is *not* written in the paper (`symbolic/output/`). |
| [`figures/`](figures/) | Self-contained scripts reproducing Figures 2-7 (Fokker--Planck solvers, the Monte-Carlo-vs-FP comparison, and the Ornstein--Uhlenbeck coefficient-verification of Appendix D). |
| [`regularization-exactness/`](regularization-exactness/) | A standalone mathematical note proving that the b90 close-encounter cutoff (eqn:logLambda) is **exact** -- no arbitrary constant -- plus the numerical scripts that verify each step of the proof. |
| [`applications/`](applications/) | Diffusion-time calculators for the astrophysical applications of Sec. V (ULDM, dark-matter substructure, ISM, GW backgrounds), with a regression test reproducing every number quoted in the paper. |

## Quick start

```bash
pip install -r requirements.txt   # numpy, scipy, sympy, mpmath, matplotlib

# verify all analytical results (~12 min; every check must PASS)
python symbolic/run_all.py

# reproduce the figures (see figures/README.md for runtimes; --fast for smoke runs)
python figures/fig2_adiabatic_eccentricity.py
python figures/fig3_whitenoise_ae.py
python figures/fig4_pointmass_ae.py
python figures/fig5_Td_vs_R.py
python figures/fig6_fp_vs_mc.py
python figures/fig7_coefficient_convergence.py

# reproduce the quoted diffusion times and applicability bounds
python applications/reproduce_paper_values.py

# the b90-exactness proof and its verification
(cd regularization-exactness && latexmk -pdf regularization-exactness.tex)
python regularization-exactness/exact_deflection_check.py
python regularization-exactness/three_body_check.py
python regularization-exactness/deflection_map_check.py
python regularization-exactness/cutoff_check.py
```

Tested with Python >= 3.9, numpy >= 1.24, scipy >= 1.10, sympy >= 1.12,
mpmath >= 1.3, matplotlib >= 3.7. Everything is pure Python; no compilation
or installation step is needed (scripts locate their shared modules
relative to their own path).

## Conventions

The scripts follow the paper's conventions exactly: slow variables
w = (a, e, Omega, i, omega, M0); tidal diffusion time defined by
Int <T_ij T_kl> dtau = (Gm/(a^3 T_d)) (delta delta + delta delta + delta
delta)/15 (eqn:TD-tidal); impulsive diffusion time by <Q_t/r^2> =
Gm/(15 a^3 T_d) (eqn:TD-impulsive); T_d^* = m sigma/(40 sqrt(2 pi) G m_*^2
n a) (eqn:Td*). Every hard-coded expression carries a comment naming the
equation label it implements.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).
