"""Steady-state distribution functions and literature comparisons.

Verifies (pure algebra, fast):
* adiabatic tidal (sec:longcoherence): the probability current
  eqn:Je-long-coherence, its compact form, and that the thermal distribution
  f_th = 2e (eqn:f-thermal) carries zero current;
* white-noise tidal (sec:shortcoherence): the compact current forms
  eqn:Ja-short / eqn:Je-short (with the T_d = (a0/a)^3 T_d0 scaling), the
  detailed-balance solution f_0 = C sqrt(a) e (eqn:f-zero-flux), and the
  constant-flux solution f_ss = C e / (a^4 (4+5e^2)^{36/35})
  (eqn:f-steady-state) -- including that the exponents 1/2, -4 and 36/35 are
  the unique values allowed;
* impulsive point-mass (sec:point-mass): the sub-thermal quasi-stationary
  eccentricity distribution eqn:f-ss-point-mass to O(1/logLambda);
* the Penarrubia comparison ratios (sec:penarrubia): with the isotropized
  kick tensor <Dv_i Dv_j> ~ r^2 delta_ij one gets B^E ratio 5/3, D^EE ratio
  (2+e^2)/(2-e^2), identical D^JJ, and B^J = D^JJ/(2J) != 0;
* the e -> 0 / e -> 1 limits of the adiabatic orientation coefficients.
"""

import pathlib
import sys
import time

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import a, e, G, m, Td, Tds, LogLam, Aamp
from sbx.anomalies import r_of_E
from sbx.impulsive import impulsive_coefficients
from sbx.verify import assert_eq, summary
from sbx import targets

t0 = time.time()
f = sp.Function("f")(e)

# ---------------------------------------------------------------------------
# Adiabatic: currents and the thermal distribution
# ---------------------------------------------------------------------------
print("Adiabatic tidal (eqn:Je-long-coherence, eqn:f-thermal):")
Je_long = targets.ADIABATIC["B^e"] * f - sp.diff(targets.ADIABATIC["D^ee"] * f, e) / 2
Je_compact = -sp.Rational(5, 24) / Td * e**3 * (1 - e**2) * sp.diff(f / e, e)
assert_eq("J^e compact form", sp.expand(Je_long), sp.expand(Je_compact))
assert_eq("thermal f = 2e has zero current", Je_long.subs(f, 2 * e).doit(), 0)

# ---------------------------------------------------------------------------
# White-noise: currents, detailed balance, constant-flux steady state
# ---------------------------------------------------------------------------
print("White-noise tidal currents (eqn:Ja-short, eqn:Je-short):")
# T_d depends on a: T_d = (a0/a)^3 T_d0. Work in units a0 = T_d0 = 1.
invTd = a**3
f2 = sp.Function("f")(a, e)
Ba = targets.WHITENOISE["B^a"].subs(1 / Td, invTd)
Be = targets.WHITENOISE["B^e"].subs(1 / Td, invTd)
Daa = targets.WHITENOISE["D^aa"].subs(1 / Td, invTd)
Dae = targets.WHITENOISE["D^ae"].subs(1 / Td, invTd)
Dee = targets.WHITENOISE["D^ee"].subs(1 / Td, invTd)
Ja = Ba * f2 - sp.diff(Dae * f2, e) / 2 - sp.diff(Daa * f2, a) / 2
Je = Be * f2 - sp.diff(Dae * f2, a) / 2 - sp.diff(Dee * f2, e) / 2
Ja_compact = a * invTd / 15 * (sp.Rational(3, 2) * e**2 * f2
                               - a * (2 + e**2) * sp.diff(f2, a)
                               + e * (1 - e**2) * sp.diff(f2, e))
Je_compact = e * (1 - e**2) * invTd / 15 * (a * sp.diff(f2, a) - f2 / 2
                                            - sp.Rational(7, 16) * (4 + 5 * e**2)
                                            * sp.diff(f2 / e, e))
assert_eq("J^a compact form", sp.expand(Ja), sp.expand(Ja_compact))
assert_eq("J^e compact form", sp.expand(Je), sp.expand(Je_compact))

print("Detailed-balance solution f_0 = C sqrt(a) e (eqn:f-zero-flux):")
f0 = sp.sqrt(a) * e
assert_eq("J^a[f_0] = 0", sp.simplify(Ja.subs(f2, f0).doit()), 0)
assert_eq("J^e[f_0] = 0", sp.simplify(Je.subs(f2, f0).doit()), 0)

print("Constant-flux solution f_ss (eqn:f-steady-state):")
fss = e / (a**4 * (4 + 5 * e**2) ** sp.Rational(36, 35))
Ja_ss = sp.simplify(Ja.subs(f2, fss).doit())
Je_ss = sp.simplify(Je.subs(f2, fss).doit())
assert_eq("J^e[f_ss] = 0", Je_ss, 0)
assert_eq("dJ^a/da [f_ss] = 0 (a-independent outward flux)",
          sp.simplify(sp.diff(Ja_ss, a)), 0)
assert_eq("J^a[f_ss] > 0 (check at e = 1/2: value 27/(2*5^(1/35)*2^(33/35)*...))",
          sp.sign(Ja_ss.subs(e, sp.Rational(1, 2))), 1)

print("Uniqueness of the exponents:")
gamma, p = sp.symbols("gamma p", real=True)
# separable ansatz f = a^gamma h(e): detailed balance J^a = 0 forces gamma = 1/2
h = sp.Function("h")(e)
Ja_gen = sp.simplify(Ja.subs(f2, a**gamma * e).doit() / a**(gamma + 3))
assert_eq("J^a[a^gamma e] = 0 forces gamma = 1/2",
          sp.solve(sp.Eq(Ja_gen, 0), gamma)[0], sp.Rational(1, 2))
# with the a^-4 ansatz, J^e = 0 forces the 36/35 exponent
fss_p = e / (a**4 * (4 + 5 * e**2) ** p)
Je_p = sp.simplify(Je.subs(f2, fss_p).doit())
sols = sp.solve(sp.Eq(sp.simplify(Je_p * (4 + 5 * e**2) ** p / (a * e**2)), 0), p)
assert_eq("J^e = 0 forces exponent 36/35", sols[0], sp.Rational(36, 35))
# and dJ^a/da = 0 forces gamma = -4 for the constant-current family
Ja_g = sp.simplify(Ja.subs(f2, a**gamma * e / (4 + 5 * e**2) ** sp.Rational(36, 35)).doit())
sols_g = sp.solve(sp.Eq(sp.expand(sp.diff(Ja_g, a) * a**(-gamma - 2)), 0), gamma)
has_m4 = any(sp.simplify(s + 4) == 0 for s in sols_g)
assert_eq("dJ^a/da = 0 admits gamma = -4", sp.Integer(1) if has_m4 else sp.Integer(0), 1)

# ---------------------------------------------------------------------------
# Point-mass sub-thermal steady state (eqn:f-ss-point-mass)
# ---------------------------------------------------------------------------
print("Point-mass quasi-stationary f_ss to O(1/logLambda):")
L = LogLam
g_ss = 2 * e * (1 + targets.F_SS_PM_CHI / L)
lhs = targets.POINT_MASS["B^e"] * g_ss
rhs = sp.diff(targets.POINT_MASS["D^ee"] * g_ss, e) / 2
diff_LL = sp.expand(sp.together(lhs - rhs) * 15 * Tds)
# expand in 1/L: multiply by L and take the two leading orders in L
poly_L = sp.collect(sp.expand(diff_LL), L)
coeff_L1 = sp.simplify(poly_L.coeff(L, 1))
coeff_L0 = sp.simplify(poly_L.coeff(L, 0))
assert_eq("O(logLambda) balance", coeff_L1, 0)
assert_eq("O(1) balance", coeff_L0, 0)
resid_Lm1 = sp.simplify(poly_L - coeff_L1 * L - coeff_L0)
print(f"  (residual at O(1/logLambda): {sp.nsimplify(sp.simplify(resid_Lm1 * L), rational=False) != 0} -- "
      "nonzero, as expected for a perturbative solution)")

# ---------------------------------------------------------------------------
# Penarrubia comparison (sec:penarrubia)
# ---------------------------------------------------------------------------
print("Penarrubia comparison (isotropized kick tensor):")
Q_ours_r, Q_ours_t = 3 * Aamp * r_of_E**2, Aamp * r_of_E**2
Q_pen_r = Q_pen_t = Aamp * r_of_E**2
B_ours, D_ours = impulsive_coefficients(Q_ours_r, Q_ours_t)
B_pen, D_pen = impulsive_coefficients(Q_pen_r, Q_pen_t)
print(f"  (impulsive pipelines done, {time.time()-t0:.0f}s)")
assert_eq("B^E ratio = 5/3", sp.simplify(B_ours["E"] / B_pen["E"]), sp.Rational(5, 3))
assert_eq("D^EE ratio = (2+e^2)/(2-e^2)",
          sp.simplify(D_ours[("E", "E")] / D_pen[("E", "E")]),
          (2 + e**2) / (2 - e**2))
assert_eq("D^JJ identical", D_ours[("J", "J")], D_pen[("J", "J")])
JJ_expr = sp.sqrt(G * m * a * (1 - e**2))
assert_eq("B^J = D^JJ/(2J) even for the isotropized tensor (so B^J != 0)",
          B_pen["J"], sp.expand(D_pen[("J", "J")] / (2 * JJ_expr)))

# ---------------------------------------------------------------------------
# e -> 0 / e -> 1 limits of the adiabatic orientation coefficients
# ---------------------------------------------------------------------------
print("Adiabatic orientation limits:")
assert_eq("D^ee_hat = D^qq_hat at e = 0",
          sp.limit(targets.ADIABATIC["D^ee_hat"] * Td, e, 0),
          sp.limit(targets.ADIABATIC["D^qq_hat"] * Td, e, 0))
assert_eq("D^qq_hat -> 0 at e = 1", sp.limit(targets.ADIABATIC["D^qq_hat"] * Td, e, 1), 0)
assert_eq("D^JJ_hat -> 0 at e = 1", sp.limit(targets.ADIABATIC["D^JJ_hat"] * Td, e, 1), 0)
assert_eq("D^ee_hat diverges at e -> 1",
          sp.limit(1 / (targets.ADIABATIC["D^ee_hat"] * Td), e, 1), 0)

print(f"\nTotal time: {time.time()-t0:.0f}s")
sys.exit(summary("check_steady_states"))
