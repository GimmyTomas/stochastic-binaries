"""Very large perturbers: the impulsive machinery must recover the white-noise
tidal results (sec:large-perturbers), and the density-fluctuation route must
give the same diffusion time (sec:density-fluctuations).

Checks performed:
1. For a Gaussian perturber with R >> a, the Fourier-space integrals
   (eqn:Yperp+Yparallel, eqn:Yperp-Yparallel with small-argument Bessel
   kernels) give Y_perp + Y_par = pi r_perp^2/(2R^2), Y_perp - Y_par =
   pi r_perp^2/(4R^2), whence (eqn:Qr-large, eqn:Qt-large)
   Q_r = 3 Q_t = (4/5) sqrt(2 pi) G^2 m_*^2 n r^2 / (sigma R^2).
2. The draft's orbit-average identities <r^2>, <r^4>, <3 r~^2 v~_r^2 +
   r~^2 v~_phi^2> quoted in sec:large-perturbers.
3. Substituting Q_r = 3 A r^2, Q_t = A r^2 with A = Gm/(15 a^3 Td)
   (eqn:TD-impulsive) into the impulsive formalism reproduces EXACTLY all
   white-noise tidal coefficients: the (a,e) sector, the (E,J) sector, the
   full Euler-angle sector of Appendix B, and the body frame.
4. eqn:Td-large: T_d = sigma R^2 m / (4 sqrt(2 pi) G m_*^2 n a^3).
5. The independent power-spectrum route (eqn:Td-Prho with P_rho =
   m_*^2 n rho~(p)^2 exp(-p^2 sigma^2 tau^2/2)) gives the SAME T_d
   (eqn:Td-tidal-R), completing the consistency triangle.
"""

import pathlib
import sys
import time

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import (a, e, G, m, E, U, Om, inc, om, M0, Td, SLOW_VARS,
                         sigma, mstar, nden, Rs, SQ)
from sbx.averaging import orbit_average
from sbx.anomalies import r_of_E
from sbx.impulsive import impulsive_coefficients
from sbx.frames import body_frame
from sbx.verify import assert_eq, summary
from sbx import targets

t0 = time.time()

# ---------------------------------------------------------------------------
# 1. Gaussian perturber, R >> a: kick covariance from the Fourier route
# ---------------------------------------------------------------------------
print("Kick covariance for a large Gaussian perturber (Fourier route):")
k, rperp, th = sp.symbols("k r_perp theta", positive=True)
rho_k = sp.exp(-k**2 * Rs**2 / 2)
# Small-argument kernels: 1 - J0(x) ~ x^2/4, J2(x) ~ x^2/8
Ysum = 4 * sp.pi * sp.integrate(rho_k**2 * (k * rperp) ** 2 / 4 / k, (k, 0, sp.oo))
Ydiff = 4 * sp.pi * sp.integrate(rho_k**2 * (k * rperp) ** 2 / 8 / k, (k, 0, sp.oo))
assert_eq("Y_perp + Y_par = pi r_perp^2/(2 R^2)", Ysum, sp.pi * rperp**2 / (2 * Rs**2))
assert_eq("Y_perp - Y_par = pi r_perp^2/(4 R^2)", Ydiff, sp.pi * rperp**2 / (4 * Rs**2))

# theta-integrals (eqn:Qr, eqn:Q) with <V^-1> = sqrt(2/pi)/sigma
Yperp = (Ysum + Ydiff) / 2
Vm1 = sp.sqrt(2 / sp.pi) / sigma
r_sym = sp.Symbol("r", positive=True)
Qr_large = 2 * G**2 * mstar**2 * nden * Vm1 * sp.integrate(
    Yperp.subs(rperp, r_sym * sp.sin(th)) * sp.sin(th) ** 3, (th, 0, sp.pi))
Qtot_large = 2 * G**2 * mstar**2 * nden * Vm1 * sp.integrate(
    Ysum.subs(rperp, r_sym * sp.sin(th)) * sp.sin(th), (th, 0, sp.pi))
Qt_large = sp.simplify((Qtot_large - Qr_large) / 2)
tgt_Qr = 4 * sp.sqrt(2 * sp.pi) * G**2 * mstar**2 * nden * r_sym**2 / (5 * sigma * Rs**2)
assert_eq("Q_r (eqn:Qr-large)", Qr_large, tgt_Qr)
assert_eq("Q_t = Q_r/3 (eqn:Qt-large)", Qt_large, tgt_Qr / 3)

# ---------------------------------------------------------------------------
# 2. Orbit-average identities quoted in sec:large-perturbers
# ---------------------------------------------------------------------------
print("Orbit-average identities:")
vr_t = e * sp.sin(E) / U
vphi_t = SQ / U
assert_eq("<r^2>", orbit_average(r_of_E**2), a**2 * (2 + 3 * e**2) / 2)
assert_eq("<r^4>", orbit_average(r_of_E**4), a**4 * (8 + 40 * e**2 + 15 * e**4) / 8)
assert_eq("<3 r~^2 v~_r^2 + r~^2 v~_phi^2>",
          orbit_average(3 * U**2 * vr_t**2 + U**2 * vphi_t**2), (2 + e**2) / 2)

# ---------------------------------------------------------------------------
# 3. Q_r = 3 A r^2, Q_t = A r^2 recovers ALL white-noise tidal coefficients
# ---------------------------------------------------------------------------
print("Impulsive formalism with tidal-limit Q (A = Gm/(15 a^3 Td)) ...")
Aamp_expr = G * m / (15 * a**3 * Td)
Qr_tid = 3 * Aamp_expr * r_of_E**2
Qt_tid = Aamp_expr * r_of_E**2
B, D = impulsive_coefficients(Qr_tid, Qt_tid)
print(f"  done ({time.time()-t0:.0f}s)")

print("(a,e) sector -> eqn:Ba-short-coherence .. eqn:Dee-short-coherence:")
assert_eq("B^a", B[a], targets.WHITENOISE["B^a"])
assert_eq("B^e", B[e], targets.WHITENOISE["B^e"])
assert_eq("D^aa", D[(a, a)], targets.WHITENOISE["D^aa"])
assert_eq("D^ae", D[(a, e)], targets.WHITENOISE["D^ae"])
assert_eq("D^ee", D[(e, e)], targets.WHITENOISE["D^ee"])
assert_eq("B^M0", B[M0], targets.WHITENOISE["B^M0"])

print("(E,J) sector -> eqn:BE-short .. eqn:DJJ-short:")
assert_eq("B^E", B["E"], targets.WHITENOISE["B^E"])
assert_eq("B^J", B["J"], targets.WHITENOISE["B^J"])
assert_eq("D^EE", D[("E", "E")], targets.WHITENOISE["D^EE"])
assert_eq("D^EJ", D[("E", "J")], targets.WHITENOISE["D^EJ"])
assert_eq("D^JJ", D[("J", "J")], targets.WHITENOISE["D^JJ"])

print("Euler-angle sector -> Appendix B (white-noise block):")
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

print("Body frame -> eqn:Dhatehate-short .. eqn:DM0M0-short:")
B_eul = [B[Om], B[inc], B[om]]
D_eul = sp.Matrix(3, 3, lambda p, q: D[((Om, inc, om)[p], (Om, inc, om)[q])])
D_eulM0 = [D[(Om, M0)], D[(inc, M0)], D[(om, M0)]]
B_hat, D_hat, D_hatM0 = body_frame(B_eul, D_eul, D_eulM0)
assert_eq("B^ehat = B^qhat = B^Jhat = 0",
          sum(sp.Abs(sp.simplify(x)) for x in B_hat), 0)
assert_eq("D^ee_hat", D_hat[0, 0], targets.WHITENOISE["D^ee_hat"])
assert_eq("D^qq_hat", D_hat[1, 1], targets.WHITENOISE["D^qq_hat"])
assert_eq("D^JJ_hat", D_hat[2, 2], targets.WHITENOISE["D^JJ_hat"])
assert_eq("D^JM0_hat", D_hatM0[2], targets.WHITENOISE["D^JM0_hat"])

# ---------------------------------------------------------------------------
# 4-5. The diffusion time: eqn:Td-large and the P_rho route (eqn:Td-Prho)
# ---------------------------------------------------------------------------
print("Diffusion time:")
A_gauss = sp.simplify(Qt_large / r_sym**2)
Td_large = sp.simplify(G * m / (15 * a**3) / A_gauss)
tgt_Td = sigma * Rs**2 * m / (4 * sp.sqrt(2 * sp.pi) * G * mstar**2 * nden * a**3)
assert_eq("T_d (eqn:Td-large)", Td_large, tgt_Td)

# Independent route: P_rho(p, tau) = m_*^2 n rho~(p)^2 exp(-p^2 s^2 t^2/2)
p, tau = sp.symbols("p tau", positive=True)
P_rho = mstar**2 * nden * sp.exp(-p**2 * Rs**2) * sp.exp(-p**2 * sigma**2 * tau**2 / 2)
inner = sp.integrate(P_rho, (tau, -sp.oo, sp.oo))
Jint = sp.integrate(p**2 / (2 * sp.pi**2) * inner, (p, 0, sp.oo))
Td_Prho = sp.simplify(G * m / ((4 * sp.pi * G) ** 2 * a**3) / Jint)
assert_eq("T_d from P_rho (eqn:Td-Prho -> eqn:Td-tidal-R)", Td_Prho, tgt_Td)

print(f"\nTotal time: {time.time()-t0:.0f}s")
sys.exit(summary("check_impulsive_tidal_limit"))
