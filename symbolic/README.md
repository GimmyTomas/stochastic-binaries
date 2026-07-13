# Symbolic verification of every analytical result

SymPy scripts that re-derive, from first principles, every drift and
diffusion coefficient and every steady-state distribution of the paper, and
check them against the published expressions **exactly** (symbolic equality,
not numerics). Each check is tagged in the output:

* `[SYM ]` — exact symbolic identity;
* `[NUM ]` — verified numerically at 30–50 digits (used only where a SymPy
  reduction is impractical, e.g. some Bessel-integral identities);
* `[FAIL]` — mismatch (never expected).

## Run everything

```bash
python run_all.py          # full suite, ~12 min, exit code 0 iff all pass
python run_all.py --fast   # skips the three heaviest scripts, ~4 min
```

> **Known pitfall.** sympy 1.12 with the optional gmpy2 accelerator installed
> (Anaconda default) hangs indefinitely on `check_impulsive_point_mass.py`.
> Use sympy >= 1.13, or set `SYMPY_GROUND_TYPES=python`. See the top-level
> README for the tested-version matrix.

## Scripts

| script | verifies | time |
|---|---|---|
| `tests_averaging.py` | every orbit-average moment table entry vs 50-digit quadrature | 10 s |
| `check_form_factors.py` | Table I: mass norms, I(b), rho~(k), and the Hankel relation eqn:rho(k), for all four extended profiles | 30 s |
| `check_steady_states.py` | thermal 2e, f_0 = sqrt(a) e, f_ss with the 36/35 exponent (incl. uniqueness), sub-thermal point-mass f_ss with the full logLambda(a,e) (J^e and d_a J^a vanish at the two leading orders in logLambda_0, (2e/3) logLambda_0 outward flux, cross-term a-independence, g/(a^2 logLambda) ansatz expansion, integrating-factor a_0-slice consistency), Penarrubia ratios 5/3 and (2+e^2)/(2-e^2), B^J = D^JJ/2J, e->0/1 limits | 40 s |
| `check_kick_covariance.py` | Maxwellian moments incl. <V^-1 log V^4>, the Y_perp -/+ Y_par integrals (point-mass, Gaussian-regulated, b90-regulated), the exact -2/3 and -8/3 constants of Q_r, Q_t, Parseval consistency | 2 min |
| `check_tidal_adiabatic.py` | sec:adiabatic: all coefficients in (a,e), (E,J), Euler angles (App. B) and the body frame; rotation-invariance lemmas | 40 s |
| `check_tidal_whitenoise.py` | sec:white-noise: same four parametrizations; (E,J) also via the first-principles route of App. C; guard test proving the fixed-phase dE/de, dE/dM0 terms (eqn:dEdedM0) are essential; e->0 degeneracy of the (Jhat, M0) block | 1 min |
| `check_tidal_traceless_gw.py` | sec:lit-gw-background: the traceless-correlator coefficients AND the claim that all others are unchanged | 3 min |
| `check_impulsive_general.py` | eqn:Ba-impulsive .. eqn:DM0M0-impulsive as exact functionals of generic Q_r(r), Q_t(r); derives the Euler-angle sector **not** in the paper -> `output/impulsive-euler-coefficients.tex` | 40 s |
| `check_impulsive_tidal_limit.py` | sec:large-perturbers: Q = A r^2 (3,1,1) reproduces ALL white-noise tidal coefficients; eqn:Td-large; the independent P_rho route eqn:Td-Prho | 40 s |
| `check_impulsive_point_mass.py` | sec:point-mass: all ten closed forms, the (E,J) block, logLambda (eqn:logLambda), the T_d <-> T_d^* relation (eqn:Td*), limits, leading logs | 4 min |

## How it works (`sbx/` package)

* `averaging.py` — orbit averages (eqn:orbit-average) are evaluated by
  table lookup: every expression is reduced to monomials
  `cos^c E sin^s E (1-e cos E)^q [log(1-e cos E)]`, whose averages have exact
  closed forms (Wallis integrals, a derivative recursion for negative powers,
  and classical Fourier-series products for the log moments). Every table
  entry is unit-tested at 50 digits.
* `anomalies.py` — the fixed-fast-phase derivative (eqn:dEdedM0), the single
  chokepoint through which all drift computations go.
* `gauss.py`, `correlators.py`, `frames.py`, `ito.py` — Gauss's equations,
  the isotropic correlator contractions, body-frame transforms (eqn:change-of-basis-matrix),
  and the Ito change of variables (eqn:B-transform / eqn:D-transform).
* `targets.py` — every published expression, transcribed verbatim and keyed
  by its LaTeX `\label`.
* `verify.py` — canonicalization (w = sqrt(1-e^2) reduction, log-atom basis)
  plus an automatic high-precision numeric fallback, so a SymPy
  simplification weakness can never masquerade as (or hide) a physics error.
