"""Regression test: reproduce every T_d value and applicability bound quoted
in Sec. V of the paper (all quoted to 2 significant figures; tolerance 5%).

Run:  python reproduce_paper_values.py     (exit code 0 iff all pass)
"""

import sys

import numpy as np
from scipy.optimize import brentq

from td_calculators import (G_PC, UNIT_GYR, td_uldm, td_tidal_perturbers,
                            td_pointmass_perturbers, td_ism, _selfcheck)

FAIL = 0


def check(name, got, want, tol=0.05):
    global FAIL
    rel = abs(got / want - 1)
    ok = rel < tol
    FAIL += not ok
    print(f"  {name:58s} {got:11.4g}  (paper: {want:g}, "
          f"{'PASS' if ok else 'FAIL'} @ {tol:.0%})")


def orbital_period_gyr(a_pc, m_msun):
    return 2 * np.pi * np.sqrt(a_pc**3 / (G_PC * m_msun)) * UNIT_GYR


print("Internal consistency:")
if not _selfcheck():
    FAIL += 1

print("Diffusion times (Sec. V):")
check("ULDM (eqn:Td-uldm) [Gyr]", td_uldm(1, 1e-21, 100, 1, 0.1), 3.8)
check("DM subhaloes, tidal (eqn:Td-large-substructures) [Gyr]",
      td_tidal_perturbers(1, 1e3, 100, 0.1, 0.026, 0.1), 0.87)
check("MACHOs/PBHs, point-mass (eqn:Td-small-substructures) [Gyr]",
      td_pointmass_perturbers(1, 1, 100, 0.026, 0.1, logLam=25), 3.5)
check("ISM, solar neighbourhood R8 (eqn:Td-ism) [Gyr]",
      td_ism(1, 0.1, "R8"), 45)
check("ISM, inner-galaxy LGR4 [Gyr]", td_ism(1, 0.1, "LGR4"), 1.6)

print("Slow-evolution bounds T = T_d (sec:slow-evolution):")
# eqn:T<<Td-large with reference values (R/a = 1, sigma = 100 km/s, m = 1,
# rho = 1 Msun/pc^3, m_* = 1): quoted prefactor 3.6 pc
a_large = brentq(lambda a: orbital_period_gyr(a, 1)
                 - td_tidal_perturbers(1, 1, 100, a, 1.0, a), 1e-4, 1e3)
check("large perturbers: a with T = T_d [pc]", a_large, 3.6)
# eqn:T<<Td-small (logLambda = 25): quoted prefactor 0.39 pc
a_small = brentq(lambda a: orbital_period_gyr(a, 1)
                 - td_pointmass_perturbers(1, 1, 100, 1.0, a, logLam=25),
                 1e-5, 1e3)
check("small perturbers: a with T = T_d [pc]", a_small, 0.39)

print("Shot-noise bounds 1/(n a^2 sigma) = T_d (sec:shot-noise):")
# eqn:poisson-large (R = a): a = 4 sqrt(2 pi) G m_*^2/(sigma^2 m); quoted 4.3e-6 pc
a_shot_large = 4 * np.sqrt(2 * np.pi) * G_PC * 1**2 / (100**2 * 1)
check("large perturbers (R = a) [pc]", a_shot_large, 4.3e-6)
# small perturbers: a = 40 sqrt(2 pi) G m_*^2 logLam/(sigma^2 m); quoted 1.1e-3 pc
a_shot_small = 40 * np.sqrt(2 * np.pi) * G_PC * 1**2 * 25 / (100**2 * 1)
check("small perturbers (logLam = 25) [pc]", a_shot_small, 1.1e-3)

print(f"\n{FAIL} failures")
sys.exit(1 if FAIL else 0)
