# Diffusion-time calculators (Sec. V applications)

* `td_calculators.py` — ready-to-use functions returning T_d in Gyr for:
  ultralight dark matter (eqn:Td-uldm), extended dark-matter substructure
  (eqn:Td-tidal-R / eqn:Td-large-gaia), point-mass perturbers
  (eqn:Td* / eqn:Td-small-gaia), the interstellar medium (R8 and LGR4
  TIGRESS-NCR models, sec:ism), a stochastic gravitational-wave background
  (sec:blas), and a generic density power spectrum (eqn:Td-Prho, numeric).
  Running the module prints the paper's reference values.

* `reproduce_paper_values.py` — regression test: reproduces every T_d value
  and every applicability bound quoted in Sec. V (3.8 Gyr ULDM; 0.87 / 3.5
  Gyr substructure; 45 / 1.6 Gyr ISM; the slow-evolution bounds 3.6 pc and
  0.39 pc; the shot-noise bounds 4.3e-6 pc and 1.1e-3 pc). Exit code 0 iff
  everything matches to better than 5% (the values are quoted to 2
  significant figures).
