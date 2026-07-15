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

> **Note.** The entire contents of this repository were compiled by
> [Claude](https://www.anthropic.com/claude) (Anthropic) starting from the
> information and results in the paper.

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

Requirement floors (all verified): Python >= 3.9, numpy >= 1.24 (1.x and
2.x both work), scipy >= 1.10, sympy >= 1.13, mpmath >= 1.3,
matplotlib >= 3.7. Tested with Python 3.9.18 (numpy 1.26, scipy 1.11,
matplotlib 3.8) and Python 3.12.1 (numpy 2.4, scipy 1.17, matplotlib 3.10),
with sympy 1.12, 1.13.3 and 1.14. Everything is pure Python; no compilation
or installation step is needed (scripts locate their shared modules
relative to their own path).

> **Known pitfall.** sympy 1.12 also works, *except* when the optional
> accelerator [gmpy2](https://pypi.org/project/gmpy2/) is installed (Anaconda
> base environments ship gmpy2, and sympy silently adopts it as its integer
> "ground types"): with sympy 1.12 + gmpy2 2.1.2,
> `symbolic/check_impulsive_point_mass.py` grinds without terminating (we
> killed it after 6.5 h; every other script is unaffected). sympy >= 1.13 is
> immune (verified against gmpy2 2.1.2), and setting the environment variable
> `SYMPY_GROUND_TYPES=python` also fixes sympy 1.12 (verified, ~3.5 min).

## Conventions

The scripts follow the paper's conventions exactly: slow variables
w = (a, e, Omega, i, omega, M0); tidal diffusion time defined by
Int <T_ij T_kl> dtau = (Gm/(a^3 T_d)) (delta delta + delta delta + delta
delta)/15 (eqn:Td-tidal); impulsive diffusion time by <Q_t/r^2> =
Gm/(15 a^3 T_d) (eqn:Td-impulsive); T_d^* = m sigma/(40 sqrt(2 pi) G m_*^2
n a) (eqn:Td*). Every hard-coded expression carries a comment naming the
equation label it implements.

## Citation

If you use this code, please cite the paper (arXiv number to be added):

```bibtex
@article{Caputo:2026fuj,
    author = "Caputo, Andrea and Tomaselli, Giovanni Maria and Hamilton, Chris",
    title = "{Evolution of Binaries Under Stochastic Perturbations}",
    eprint = "2607.12011",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.GA",
    month = "7",
    year = "2026"
}
```
