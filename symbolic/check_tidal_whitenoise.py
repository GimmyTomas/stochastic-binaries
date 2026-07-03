"""White-noise tidal perturbations (sec:shortcoherence): verify every drift
and diffusion coefficient of the paper from first principles.

Pipeline (eqn:Deltaw2, eqn:Deltaww2): contract the force tensors at fixed
orbital phase FIRST, orbit-average LAST. The drift requires the fixed-phase
derivatives dE/de = sinE/(1-e cosE), dE/dM0 = 1/(1-e cosE) (eqn:dEdedM0);
a guard test proves that omitting them gives the WRONG B^e, so this subtlety
cannot silently regress.

Checked parametrizations: (a, e); (E, J) both via the Ito transformation and
via the independent first-principles route of Appendix C; the Euler angles
(Omega, i, omega, M0) of Appendix B; and the body frame (ehat, qhat, Jhat, M0),
including the degenerate e -> 0 structure of the (Jhat, M0) block.
"""

import pathlib
import sys
import time

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import a, e, G, m, Td, Om, inc, om, M0, E, U, SLOW_VARS, SQ
from sbx.averaging import orbit_average
from sbx.anomalies import cosphi, sinphi, r_of_E
from sbx.pipelines import whitenoise_coefficients, PREF
from sbx.correlators import contract
from sbx.frames import body_frame
from sbx.ito import ito_transform
from sbx.verify import assert_eq, summary
from sbx import targets

t0 = time.time()

print("Computing white-noise coefficients (contract-then-average)...")
B, D = whitenoise_coefficients("traceful")
print(f"  done ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# (a, e) sector
# ---------------------------------------------------------------------------
print("(a,e) sector (eqn:Ba-short-coherence .. eqn:Dee-short-coherence):")
assert_eq("B^a", B[a], targets.WHITENOISE["B^a"])
assert_eq("B^e", B[e], targets.WHITENOISE["B^e"])
assert_eq("D^aa", D[(a, a)], targets.WHITENOISE["D^aa"])
assert_eq("D^ae", D[(a, e)], targets.WHITENOISE["D^ae"])
assert_eq("D^ee", D[(e, e)], targets.WHITENOISE["D^ee"])
assert_eq("B^M0", B[M0], targets.WHITENOISE["B^M0"])

print("Block structure: (a,e) decouples; M0 couples only to omega:")
offblock = [D[(a, v)] for v in (Om, inc, om, M0)] + [D[(e, v)] for v in (Om, inc, om, M0)]
assert_eq("D^{a,rest} = D^{e,rest} = 0", sum(sp.Abs(sp.simplify(x)) for x in offblock), 0)
assert_eq("D^{Omega M0} = 0", D[(Om, M0)], 0)
assert_eq("D^{i M0} = 0", D[(inc, M0)], 0)

# ---------------------------------------------------------------------------
# Guard test: the fixed-phase derivative terms are essential
# ---------------------------------------------------------------------------
print("Guard test: naive derivative (no dE/de, dE/dM0 terms) must FAIL:")
B_naive, _ = whitenoise_coefficients("traceful", naive_drift=True)
resid = sp.simplify(B_naive[e] - targets.WHITENOISE["B^e"])
if resid == 0:
    assert_eq("naive B^e differs from eqn:Ba-short-coherence-block", 1, 0)  # force a FAIL
else:
    print(f"  [SYM ] naive B^e differs (residual {resid}) -- fixed-phase terms matter")

# ---------------------------------------------------------------------------
# (E, J) sector: Ito route AND the first-principles route of Appendix C
# ---------------------------------------------------------------------------
print("(E,J) sector via Ito transform (eqn:BE-short .. eqn:DJJ-short):")
EE_expr = -G * m / (2 * a)
JJ_expr = sp.sqrt(G * m * a * (1 - e**2))
B_EJ, D_EJ = ito_transform([a, e], [EE_expr, JJ_expr],
                           [B[a], B[e]],
                           sp.Matrix([[D[(a, a)], D[(a, e)]], [D[(a, e)], D[(e, e)]]]))
assert_eq("B^E (Ito)", B_EJ[0], targets.WHITENOISE["B^E"])
assert_eq("B^J (Ito)", B_EJ[1], targets.WHITENOISE["B^J"])
assert_eq("D^EE (Ito)", D_EJ[0, 0], targets.WHITENOISE["D^EE"])
assert_eq("D^EJ (Ito)", D_EJ[0, 1], targets.WHITENOISE["D^EJ"])
assert_eq("D^JJ (Ito)", D_EJ[1, 1], targets.WHITENOISE["D^JJ"])

print("(E,J) sector, first-principles route (Appendix C):")
# Perifocal-frame position and velocity of the Kepler orbit (U-form):
rvec = a * sp.Matrix([sp.cos(E) - e, SQ * sp.sin(E), 0])
vvec = sp.sqrt(G * m / a) * sp.Matrix([-sp.sin(E) / U, SQ * sp.cos(E) / U, 0])
rhat = sp.Matrix([cosphi, sinphi, 0])
phihat = sp.Matrix([-sinphi, cosphi, 0])
r = r_of_E
# With the tidal acceleration a_i = -T_ij r_j:
#   Edot = v . a          => g^E_ij = -v_i r_j
#   Jdot = (r x a)_z      => g^J_ij = -r^2 phihat_i rhat_j
# (the two minus signs cancel in every product below, and each g appears
# quadratically or in the EJ product, so we drop them consistently).
gE_mat = vvec * rvec.T
gJ_mat = r**2 * (phihat * rhat.T)
# Appendix-C drift: B^E = (1/2) <r_j r_k ONE_ikij> Gm/(15 a^3 Td)
#                        = <r^2>/6 * Gm/(a^3 Td)   since ONE_ikij = 5 d_jk.
BE_fp = orbit_average(sp.expand(rvec.dot(rvec))) / 6 * (G * m / (a**3 * Td))
assert_eq("B^E (first principles)", BE_fp, targets.WHITENOISE["B^E"])
DEE_fp = orbit_average(sp.expand(contract(gE_mat, gE_mat))) * PREF
assert_eq("D^EE (first principles)", DEE_fp, targets.WHITENOISE["D^EE"])
DEJ_fp = orbit_average(sp.expand(contract(gE_mat, gJ_mat))) * PREF
assert_eq("D^EJ (first principles)", DEJ_fp, targets.WHITENOISE["D^EJ"])
DJJ_fp = orbit_average(sp.expand(contract(gJ_mat, gJ_mat))) * PREF
assert_eq("D^JJ (first principles)", DJJ_fp, targets.WHITENOISE["D^JJ"])
# B^{J^2} = 2 r^4 * Gm/(15 a^3 Td)  and  B^J = (B^{J^2} - D^JJ)/(2J)
BJ2_fp = orbit_average(2 * sp.expand(r**4)) * PREF
BJ_fp = sp.simplify((BJ2_fp - DJJ_fp) / (2 * JJ_expr))
assert_eq("B^J (first principles)", BJ_fp, targets.WHITENOISE["B^J"])
assert_eq("B^J = D^JJ/(2J) (App C identity)", BJ_fp, sp.simplify(DJJ_fp / (2 * JJ_expr)))

# ---------------------------------------------------------------------------
# Euler-angle sector (Appendix B, white-noise block)
# ---------------------------------------------------------------------------
print("Euler-angle sector (Appendix B, white-noise):")
assert_eq("B^Omega", B[Om], targets.WHITENOISE_EULER["B^Omega"])
assert_eq("B^i", B[inc], targets.WHITENOISE_EULER["B^i"])
assert_eq("B^omega", B[om], targets.WHITENOISE_EULER["B^omega"])
assert_eq("D^OmegaOmega", D[(Om, Om)], targets.WHITENOISE_EULER["D^OmegaOmega"])
assert_eq("D^Omegai", D[(Om, inc)], targets.WHITENOISE_EULER["D^Omegai"])
assert_eq("D^Omegaomega", D[(Om, om)], targets.WHITENOISE_EULER["D^Omegaomega"])
assert_eq("D^ii", D[(inc, inc)], targets.WHITENOISE_EULER["D^ii"])
assert_eq("D^iomega", D[(inc, om)], targets.WHITENOISE_EULER["D^iomega"])
assert_eq("D^omegaomega", D[(om, om)], targets.WHITENOISE_EULER["D^omegaomega"])
assert_eq("D^omegaM0", D[(om, M0)], targets.WHITENOISE_EULER["D^omegaM0"])
assert_eq("D^M0M0", D[(M0, M0)], targets.WHITENOISE["D^M0M0"])

# ---------------------------------------------------------------------------
# Body frame (ehat, qhat, Jhat, M0)
# ---------------------------------------------------------------------------
print("Body-frame sector (eqn:Dhatehate-short .. eqn:DM0M0-short):")
B_eul = [B[Om], B[inc], B[om]]
D_eul = sp.Matrix(3, 3, lambda p, q: D[((Om, inc, om)[p], (Om, inc, om)[q])])
D_eulM0 = [D[(Om, M0)], D[(inc, M0)], D[(om, M0)]]
B_hat, D_hat, D_hatM0 = body_frame(B_eul, D_eul, D_eulM0)
assert_eq("B^ehat = 0", B_hat[0], 0)
assert_eq("B^qhat = 0", B_hat[1], 0)
assert_eq("B^Jhat = 0", B_hat[2], 0)
assert_eq("D^ee_hat", D_hat[0, 0], targets.WHITENOISE["D^ee_hat"])
assert_eq("D^qq_hat", D_hat[1, 1], targets.WHITENOISE["D^qq_hat"])
assert_eq("D^JJ_hat", D_hat[2, 2], targets.WHITENOISE["D^JJ_hat"])
assert_eq("D^eq_hat = 0", D_hat[0, 1], 0)
assert_eq("D^eJ_hat = 0", D_hat[0, 2], 0)
assert_eq("D^qJ_hat = 0", D_hat[1, 2], 0)
assert_eq("D^eM0_hat = 0", D_hatM0[0], 0)
assert_eq("D^qM0_hat = 0", D_hatM0[1], 0)
assert_eq("D^JM0_hat", D_hatM0[2], targets.WHITENOISE["D^JM0_hat"])

# ---------------------------------------------------------------------------
# Degenerate e -> 0 limit of the (Jhat, M0) block (sec:orientation-short-coherence)
# ---------------------------------------------------------------------------
print("e -> 0 degeneracy of the (Jhat, M0) block:")
DJJh = targets.WHITENOISE["D^JJ_hat"]
DJM0h = targets.WHITENOISE["D^JM0_hat"]
DM0M0h = targets.WHITENOISE["D^M0M0"]
lim_JJ = sp.limit(sp.together(e**2 * DJJh * Td), e, 0)
lim_JM0 = sp.limit(sp.together(e**2 * DJM0h * Td), e, 0)
lim_M0M0 = sp.limit(sp.together(e**2 * DM0M0h * Td), e, 0)
assert_eq("e^2 D^JJ_hat -> 7/30", lim_JJ, sp.Rational(7, 30))
assert_eq("e^2 D^JM0_hat -> -7/30", lim_JM0, -sp.Rational(7, 30))
assert_eq("e^2 D^M0M0 -> 7/30", lim_M0M0, sp.Rational(7, 30))
assert_eq("lambda0 = M0 + omega does not diffuse at e=0 (pole cancels)",
          sp.limit(sp.together(e**2 * (DJJh + 2 * DJM0h + DM0M0h) * Td), e, 0), 0)

# Consistency of the drafts's verbal claims: e->0 and e->1 body-frame limits
print("e -> 0 / e -> 1 limits of the orientation coefficients:")
assert_eq("D^ee_hat = D^qq_hat at e=0",
          sp.limit(targets.WHITENOISE["D^ee_hat"] * Td, e, 0),
          sp.limit(targets.WHITENOISE["D^qq_hat"] * Td, e, 0))
assert_eq("D^qq_hat = D^JJ_hat at e=1",
          sp.limit(targets.WHITENOISE["D^qq_hat"] * Td, e, 1),
          sp.limit(targets.WHITENOISE["D^JJ_hat"] * Td, e, 1))

print(f"\nTotal time: {time.time()-t0:.0f}s")
sys.exit(summary("check_tidal_whitenoise"))
