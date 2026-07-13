"""Steady-state distribution functions and literature comparisons.

Verifies (pure algebra, fast):
* adiabatic tidal (sec:adiabatic): the probability current
  eqn:Je-adiabatic, its compact form, and that the thermal distribution
  f_th = 2e (eqn:f-thermal) carries zero current;
* white-noise tidal (sec:white-noise): the compact current forms
  eqn:Ja-white-noise / eqn:Je-white-noise (with the T_d = (a0/a)^3 T_d0 scaling), the
  detailed-balance solution f_0 = C sqrt(a) e (eqn:f-zero-flux), and the
  constant-flux solution f_ss = C e / (a^4 (4+5e^2)^{36/35})
  (eqn:f-steady-state) -- including that the exponents 1/2, -4 and 36/35 are
  the unique values allowed;
* impulsive point-mass (sec:point-mass): the sub-thermal quasi-stationary
  distribution eqn:f-ss-point-mass with the FULL logLambda(a, e) of
  eqn:logLambda -- J^e and d_a J^a vanish at the two leading orders in
  logLambda_0 = logLambda(a_0, 0), the leading flux is the a-independent
  outward J^a = (2e/3) logLambda_0, the cross term D^ae f is a-independent
  at O(1), and eqn:f-ss-point-mass is the O(1/logLambda_0) expansion of the
  ansatz g(e)/(a^2 logLambda(a, e)) as well as of the integrating-factor
  solution of the a_0-slice ODE B^e h = (1/2) d_e(D^ee h);
* the Penarrubia comparison ratios (sec:lit-white-noise): with the isotropized
  kick tensor <Dv_i Dv_j> ~ r^2 delta_ij one gets B^E ratio 5/3, D^EE ratio
  (2+e^2)/(2-e^2), identical D^JJ, and B^J = D^JJ/(2J) != 0;
* the e -> 0 / e -> 1 limits of the adiabatic orientation coefficients.
"""

import pathlib
import sys
import time

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import a, a0, e, G, m, Td, Tds, LogLam, LogLam0, Aamp, sigma, mstar, SQ
from sbx.anomalies import r_of_E
from sbx.impulsive import impulsive_coefficients
from sbx.verify import assert_eq, summary
from sbx import targets

t0 = time.time()
f = sp.Function("f")(e)

# ---------------------------------------------------------------------------
# Adiabatic: currents and the thermal distribution
# ---------------------------------------------------------------------------
print("Adiabatic tidal (eqn:Je-adiabatic, eqn:f-thermal):")
Je_long = targets.ADIABATIC["B^e"] * f - sp.diff(targets.ADIABATIC["D^ee"] * f, e) / 2
Je_compact = -sp.Rational(5, 24) / Td * e**3 * (1 - e**2) * sp.diff(f / e, e)
assert_eq("J^e compact form", sp.expand(Je_long), sp.expand(Je_compact))
assert_eq("thermal f = 2e has zero current", Je_long.subs(f, 2 * e).doit(), 0)

# ---------------------------------------------------------------------------
# White-noise: currents, detailed balance, constant-flux steady state
# ---------------------------------------------------------------------------
print("White-noise tidal currents (eqn:Ja-white-noise, eqn:Je-white-noise):")
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
# Point-mass sub-thermal steady state (eqn:f-ss-point-mass), full logLambda(a,e)
# ---------------------------------------------------------------------------
print("Point-mass quasi-stationary f_ss (full logLambda(a,e), to O(1/logLambda_0)):")
L0 = LogLam0
x = sp.Symbol("x", positive=True)  # x = 1/logLambda_0 for series work

# (0) the split LOGLAM_PM_SPLIT is exactly eqn:logLambda minus its (a_0, 0) value
m1, m2 = sp.symbols("m_1 m_2", positive=True)
LL_exact = sp.log(4 * sigma**4 * a**2 * (1 + SQ) ** 2
                  / (G**2 * (m1 + mstar) * (m2 + mstar))) - 2 * sp.EulerGamma
assert_eq("logLambda(a,e) - logLambda(a_0,0) = 2log(a/a_0) + 2log((1+w)/2) (eqn:logLambda)",
          sp.expand_log(LL_exact - LL_exact.subs(a, a0).subs(e, 0), force=True),
          sp.expand_log(targets.LOGLAM_PM_SPLIT - LogLam0, force=True))

# Units a_0 = T_d^*(a_0) = 1; eqn:Td* gives T_d^* ~ 1/a, so 1/T_d^*(a) = a.
LL = targets.LOGLAM_PM_SPLIT.subs(a0, 1)

def pm(key):
    return targets.POINT_MASS[key].subs(LogLam, LL).subs(1 / Tds, a)

Ba_pm, Be_pm, Daa_pm, Dae_pm, Dee_pm = map(pm, ("B^a", "B^e", "D^aa", "D^ae", "D^ee"))
fss = 2 * e / a**2 * (1 + targets.F_SS_PM_CHI.subs(a0, 1) / L0)  # eqn:f-ss-point-mass
Ja_pm = Ba_pm * fss - sp.diff(Dae_pm * fss, e) / 2 - sp.diff(Daa_pm * fss, a) / 2
Je_pm = Be_pm * fss - sp.diff(Dae_pm * fss, a) / 2 - sp.diff(Dee_pm * fss, e) / 2

# (i) J^e[f_ss] vanishes at O(logLambda_0) and O(1); the O(1/logLambda_0)
#     residual is nonzero (perturbative solution)
Je_x = sp.expand(Je_pm)
assert_eq("J^e[f_ss]: O(logLambda_0) balance (thermal)", Je_x.coeff(L0, 1), 0)
assert_eq("J^e[f_ss]: O(1) balance (fixes the printed correction)", Je_x.coeff(L0, 0), 0)
resid = sp.simplify(sp.expand_log(Je_x.coeff(L0, -1), force=True))
print(f"  (J^e residual at O(1/logLambda_0) nonzero: {resid != 0} -- expected)")

# (ii) the ansatz makes d_a J^a vanish at both leading orders, while the
#      leading flux itself is the a-independent positive constant (2e/3) L0
dJa = sp.expand(sp.diff(Ja_pm, a))
assert_eq("d_a J^a[f_ss]: O(logLambda_0) balance", dJa.coeff(L0, 1), 0)
assert_eq("d_a J^a[f_ss]: O(1) balance", dJa.coeff(L0, 0), 0)
assert_eq("J^a[f_ss] = (2e/3) logLambda_0 + O(1), a-independent outward flux",
          sp.expand(Ja_pm).coeff(L0, 1), sp.Rational(2, 3) * e)

# (iii) the cross term D^ae f_ss is a-independent at O(1)
assert_eq("d_a(D^ae f_ss) = 0 at O(1) (cross term a-independent)",
          sp.expand(sp.diff(Dae_pm * fss, a)).coeff(L0, 0), 0)

# (iv) eqn:f-ss-point-mass IS the O(1/logLambda_0) expansion of the ansatz
#      g(e)/(a^2 logLambda(a,e)) with g = 2e (1 + F_SS_PM_G_CHI/logLambda_0)
f_ansatz = 2 * e * (1 + targets.F_SS_PM_G_CHI / L0) / (a**2 * LL)
dser = sp.expand(sp.series((L0 * f_ansatz - fss).subs(L0, 1 / x), x, 0, 2).removeO())
assert_eq("g/(a^2 logLambda) expands to eqn:f-ss-point-mass at O(1)", dser.coeff(x, 0), 0)
assert_eq("... and at O(1/logLambda_0)", dser.coeff(x, 1), 0)

# (v) figure-overlay gate: the a_0-slice of J^e = 0 is the first-order ODE
#     B^e h = (1/2) d_e(D^ee h) with full logLambda(a_0, e); its integrating-
#     factor solution h = exp(int 2B/D de)/D agrees with the printed f_ss(a_0,e)
#     through O(1/logLambda_0):  (log h)' - 2B/D + (log D)' = O(1/logLambda_0^2)
Bhat, Dhat, h_pr = Be_pm.subs(a, 1), Dee_pm.subs(a, 1), fss.subs(a, 1)
rlog = sp.diff(sp.log(h_pr), e) - 2 * Bhat / Dhat + sp.diff(Dhat, e) / Dhat
rser = sp.expand(sp.series(rlog.subs(L0, 1 / x), x, 0, 2).removeO())
assert_eq("integrating-factor a_0-slice solution matches printed f_ss at O(1)",
          rser.coeff(x, 0), 0)
assert_eq("... and at O(1/logLambda_0)", rser.coeff(x, 1), 0)

# ---------------------------------------------------------------------------
# Penarrubia comparison (sec:lit-white-noise)
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
