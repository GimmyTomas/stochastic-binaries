"""Very small / pointlike perturbers (sec:point-mass): verify the closed-form
drift and diffusion coefficients eqn:Ba-point-mass .. eqn:DM0M0-point-mass,
the (E, J) block, the T_d <-> T_d^* relation (eqn:Td*), and the e -> 0 / 1
limits quoted in the draft.

The kick covariances are (sec:point-mass)

    Q_r = kappa (L(r) - 2/3),   Q_t = kappa (L(r) - 8/3),
    kappa = 8 sqrt(2 pi) G^2 m_*^2 n / (3 sigma),
    L(r)  = L_a + 2 log(1 - e cos E),

where the r-independent constant L_a equals
log(16 sigma^4 a^2/(G^2 (m1+m_*)(m2+m_*))) - 2 gamma_E for point masses, or
log(a^2/R^2) + gamma_E for small Gaussian perturbers -- the algebra below is
identical for both, so L_a is kept symbolic. The effective Coulomb logarithm
of eqn:logLambda is logLambda = L_a + 2 log((1 + sqrt(1-e^2))/2), and the
normalization is 1/(15 T_d^*) = kappa a / (G m) (eqn:Td*).

All orbit averages, including the log moments <cos^n E log(1 - e cosE)> and
<log(1 - e cos E)/(1 - e cos E)^p>, are evaluated in closed form by
sbx.averaging (classical Fourier-series products; every table entry is
verified at 50 digits by tests_averaging.py).
"""

import pathlib
import sys
import time

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import (a, e, G, m, E, U, LNU, Om, inc, om, M0, Td, Tds,
                         LogLam, La, kappa, SQ)
from sbx.averaging import orbit_average
from sbx.anomalies import r_of_E
from sbx.impulsive import impulsive_coefficients
from sbx.frames import body_frame
from sbx.verify import assert_eq, summary
from sbx import targets

t0 = time.time()

# ---------------------------------------------------------------------------
# Kick covariances and normalizations
# ---------------------------------------------------------------------------
Lr = La + 2 * LNU
Qr_pm = kappa * (Lr - sp.Rational(2, 3))
Qt_pm = kappa * (Lr - sp.Rational(8, 3))

# eqn:logLambda: logLambda = L_a + 2 log((1 + sqrt(1-e^2))/2)
La_from_logLam = LogLam - 2 * sp.log((1 + SQ) / 2)
# eqn:Td*: T_d^* = m sigma/(40 sqrt(2pi) G m_*^2 n a)  <=>  kappa = Gm/(15 a T_d^*)
kappa_from_Tds = G * m / (15 * a * Tds)


def finalize(expr):
    """Substitute L_a -> logLambda form and kappa -> T_d^* normalization."""
    return expr.subs({La: La_from_logLam, kappa: kappa_from_Tds})


print("Computing point-mass coefficients (closed form)...")
B, D = impulsive_coefficients(Qr_pm, Qt_pm)
print(f"  done ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# (a, e) sector: eqn:Ba-point-mass .. eqn:Dee-point-mass
# ---------------------------------------------------------------------------
print("(a,e) sector:")
assert_eq("B^a", finalize(B[a]), targets.POINT_MASS["B^a"])
assert_eq("B^e", finalize(B[e]), targets.POINT_MASS["B^e"])
assert_eq("D^aa", finalize(D[(a, a)]), targets.POINT_MASS["D^aa"])
assert_eq("D^ae", finalize(D[(a, e)]), targets.POINT_MASS["D^ae"])
assert_eq("D^ee", finalize(D[(e, e)]), targets.POINT_MASS["D^ee"])
assert_eq("B^M0 = 0", B[M0], 0)

# ---------------------------------------------------------------------------
# (E, J) sector (sec:point-mass block)
# ---------------------------------------------------------------------------
print("(E,J) sector:")
assert_eq("B^E", finalize(B["E"]), targets.POINT_MASS["B^E"])
assert_eq("B^J", finalize(B["J"]), targets.POINT_MASS["B^J"])
assert_eq("D^EE", finalize(D[("E", "E")]), targets.POINT_MASS["D^EE"])
assert_eq("D^EJ", finalize(D[("E", "J")]), targets.POINT_MASS["D^EJ"])
assert_eq("D^JJ", finalize(D[("J", "J")]), targets.POINT_MASS["D^JJ"])

# ---------------------------------------------------------------------------
# Orientation sector: eqn:Dhatehate-impulsive .. eqn:DM0M0-point-mass
# ---------------------------------------------------------------------------
print("Body-frame orientation sector:")
B_eul = [B[Om], B[inc], B[om]]
D_eul = sp.Matrix(3, 3, lambda p, q: D[((Om, inc, om)[p], (Om, inc, om)[q])])
D_eulM0 = [D[(Om, M0)], D[(inc, M0)], D[(om, M0)]]
B_hat, D_hat, D_hatM0 = body_frame(B_eul, D_eul, D_eulM0)
assert_eq("B^ehat = B^qhat = B^Jhat = 0",
          sum(sp.Abs(sp.simplify(x)) for x in B_hat), 0)
assert_eq("D^ee_hat", finalize(D_hat[0, 0]), targets.POINT_MASS["D^ee_hat"])
assert_eq("D^qq_hat", finalize(D_hat[1, 1]), targets.POINT_MASS["D^qq_hat"])
assert_eq("D^JJ_hat", finalize(D_hat[2, 2]), targets.POINT_MASS["D^JJ_hat"])
assert_eq("D^JM0_hat", finalize(D_hatM0[2]), targets.POINT_MASS["D^JM0_hat"])
assert_eq("D^M0M0", finalize(D[(M0, M0)]), targets.POINT_MASS["D^M0M0"])

# ---------------------------------------------------------------------------
# T_d (eqn:TD-impulsive) versus T_d^* (eqn:Td*)
# ---------------------------------------------------------------------------
print("T_d relation (eqn:Td*):")
# Gm/(15 a^3 T_d) = <Q_t/r^2> (eqn:TD-impulsive), and eqn:Td* states
# T_d = T_d^* sqrt(1-e^2) / [logLam - 8/3 + 4 log(2 sqrt(1-e^2)/(1+sqrt(1-e^2)))].
avg_Qt_r2 = orbit_average(Qt_pm / r_of_E**2)
Td_from_avg = finalize(G * m / (15 * a**3 * avg_Qt_r2))   # = T_d, in terms of T_d^*
assert_eq("T_d * bracket = T_d^* sqrt(1-e^2)  (eqn:Td*)",
          sp.together(Td_from_avg * targets.TDSTAR_BRACKET), Tds * SQ)

# ---------------------------------------------------------------------------
# e -> 0 and e -> 1 limits quoted in the draft (sec:point-mass, orientation)
# ---------------------------------------------------------------------------
print("e -> 0 / e -> 1 limits:")
Dee_h = targets.POINT_MASS["D^ee_hat"] * 15 * Tds
Dqq_h = targets.POINT_MASS["D^qq_hat"] * 15 * Tds
DJJ_h = targets.POINT_MASS["D^JJ_hat"] * 15 * Tds
lim0 = LogLam / 2 - sp.Rational(4, 3)
assert_eq("D^ee_hat -> (logLam/2 - 4/3)/(15 Td*) at e=0", sp.limit(Dee_h, e, 0), lim0)
assert_eq("D^qq_hat -> (logLam/2 - 4/3)/(15 Td*) at e=0", sp.limit(Dqq_h, e, 0), lim0)
lim1 = (LogLam - 1) / 2
assert_eq("D^qq_hat -> (logLam-1)/(30 Td*) at e=1", sp.limit(Dqq_h, e, 1), lim1)
assert_eq("D^JJ_hat -> (logLam-1)/(30 Td*) at e=1", sp.limit(DJJ_h, e, 1), lim1)

# ---------------------------------------------------------------------------
# Hamilton--Modak limit: leading-log parts of the (a,e) coefficients
# ---------------------------------------------------------------------------
print("Leading-log (logLambda -> infinity) parts:")
for name, tgt_lead in [
    ("B^a", a * 7 * LogLam / (15 * Tds)),
    ("B^e", 5 * (1 - 3 * e**2) / (4 * e) * LogLam / (15 * Tds)),
    ("D^aa", a**2 * 4 * LogLam / (15 * Tds)),
    ("D^ee", sp.Rational(5, 2) * (1 - e**2) * LogLam / (15 * Tds)),
]:
    coeff = sp.expand(targets.POINT_MASS[name]).coeff(LogLam) * LogLam
    assert_eq(f"leading log of {name}", coeff, tgt_lead)

print(f"\nTotal time: {time.time()-t0:.0f}s")
sys.exit(summary("check_impulsive_point_mass"))
