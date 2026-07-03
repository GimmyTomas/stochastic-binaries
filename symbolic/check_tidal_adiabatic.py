"""Adiabatic tidal perturbations (sec:longcoherence): verify every drift and
diffusion coefficient of the paper from first principles.

Pipeline (eqn:Deltaw1, eqn:Deltaww1, eqn:Bmu, eqn:Dmunu):
1. build the tidal force tensors g^mu_ij from Gauss's equations (eqn:Fs-tidal);
2. orbit-average them (adiabatic regime: average FIRST, then contract);
3. B^mu = (1/2) d_nu(gbar^mu) : gbar^nu, D^munu = gbar^mu : gbar^nu, with the
   isotropic traceful correlator (Gm/(a^3 Td)) * ONE_ijkl/15 (eqn:TD-tidal);
4. compare with the paper in all four parametrizations:
   (a, e), (E, J), Euler angles (Omega, i, omega, M0) [Appendix B], and the
   body frame (ehat, qhat, Jhat, M0).

Omega-independence is not assumed: we prove once that
    g^mu(Omega) = Rz(Omega) g^mu(0) Rz(Omega)^T          (exact identity)
and that the correlator contraction is invariant under any joint rotation
(eqn:identity-rotation). Together these imply that all coefficients are
independent of Omega, so the remaining algebra may be done at Omega = 0
(after taking the d/dOmega drift terms symbolically).

Every check is exact (SymPy).
"""

import pathlib
import sys
import time

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import a, e, G, m, Td, Om, inc, om, M0, SLOW_VARS
from sbx.averaging import orbit_average
from sbx.gauss import g_tensor
from sbx.correlators import contract
from sbx.frames import R0, Rz, body_frame
from sbx.ito import ito_transform
from sbx.verify import assert_eq, summary
from sbx import targets

t0 = time.time()
PREF = sp.Rational(1, 15) * G * m / (a**3 * Td)

# ---------------------------------------------------------------------------
# 0. Rotation-invariance lemmas that justify working at Omega = 0
# ---------------------------------------------------------------------------
print("Rotation-invariance lemmas:")
X = sp.Matrix(3, 3, lambda i_, j_: sp.Symbol(f"x{i_}{j_}"))
Y = sp.Matrix(3, 3, lambda i_, j_: sp.Symbol(f"y{i_}{j_}"))
Rsym = Rz(Om)
lhs = contract(Rsym * X * Rsym.T, Rsym * Y * Rsym.T)
assert_eq("contract is Rz-invariant (eqn:identity-rotation)",
          sp.expand_trig(sp.expand(lhs)), sp.expand(contract(X, Y)))

R_full = R0()
conj_ok = sp.S.Zero
for var in SLOW_VARS:
    gfull = g_tensor(var, R=R_full)
    gzero = g_tensor(var, R=R_full.subs(Om, 0))
    diffmat = sp.expand(sp.expand_trig(gfull - Rsym * gzero * Rsym.T))
    conj_ok += sum(sp.Abs(sp.simplify(diffmat[i_, j_])) for i_ in range(3) for j_ in range(3))
assert_eq("g^mu(Omega) = Rz g^mu(0) Rz^T for all mu", conj_ok, 0)

# ---------------------------------------------------------------------------
# 1-2. Orbit-averaged force tensors and their slow-variable derivatives
# ---------------------------------------------------------------------------
print("Averaging force tensors and their derivatives (at Omega = 0)...")


def avg_matrix(gmat):
    return sp.Matrix(3, 3, lambda i_, j_: sp.simplify(orbit_average(gmat[i_, j_])))


gbar = {}     # averaged tensors at Omega = 0
dgbar = {}    # dgbar[(mu, nu)] = d(gbar^mu)/d(w^nu) at Omega = 0
for var in SLOW_VARS:
    gfull = g_tensor(var, R=R_full)
    g0 = gfull.subs(Om, 0)
    gbar[var] = avg_matrix(g0)
    # d/dOmega must be taken before setting Omega = 0
    dgbar[(var, Om)] = avg_matrix(sp.diff(gfull, Om).subs(Om, 0))
    # all other derivatives commute with the substitution Omega -> 0;
    # they act on the closed-form averages (adiabatic: differentiate the
    # averaged force, eqn:Deltaw1). d/de of the average includes the measure
    # term automatically because the average is a closed form in e.
    for nu in (a, e, inc, om, M0):
        dgbar[(var, nu)] = gbar[var].applyfunc(lambda x, v=nu: sp.diff(x, v))
print(f"  done ({time.time()-t0:.0f}s)")

# The draft's example components (eqn:gexy block) are quoted in the perifocal
# frame (R = identity): only the xy / yx components survive.
gbar_pf_a = avg_matrix(g_tensor(a))
gbar_pf_e = avg_matrix(g_tensor(e))
print("Example averaged components (eqn:gexy block):")
assert_eq("gbar^a_xy", gbar_pf_a[0, 1], targets.ADIABATIC["gbar^a_xy"])
assert_eq("gbar^a_yx", gbar_pf_a[1, 0], -targets.ADIABATIC["gbar^a_xy"])
# See the NOTE in sbx.targets: the exact result sits in the yx slot with a
# plus sign (the draft's eqn:gexy quotes -(5/2)... in xy, a sign/slot typo in
# the illustrative example only; all final coefficients below are unaffected).
assert_eq("gbar^e_yx", gbar_pf_e[1, 0], targets.ADIABATIC["gbar^e_yx"])
assert_eq("gbar^a other components vanish",
          sum(sp.Abs(gbar_pf_a[i_, j_]) for i_ in range(3) for j_ in range(3)
              if (i_, j_) not in ((0, 1), (1, 0))), 0)

# ---------------------------------------------------------------------------
# 3. Drift and diffusion coefficients (eqn:Bmu, eqn:Dmunu)
# ---------------------------------------------------------------------------
print("Contracting B^mu and D^{mu nu}...")
D = {}
for i_mu, mu in enumerate(SLOW_VARS):
    for nu in SLOW_VARS[i_mu:]:
        D[(mu, nu)] = sp.expand(contract(gbar[mu], gbar[nu])) * PREF
        D[(nu, mu)] = D[(mu, nu)]
B = {}
for mu in SLOW_VARS:
    val = sp.S.Zero
    for nu in SLOW_VARS:
        val += contract(dgbar[(mu, nu)], gbar[nu])
    B[mu] = sp.expand(val / 2) * PREF
print(f"  done ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# 4a. (a, e) sector
# ---------------------------------------------------------------------------
print("(a,e) sector (eqn:Ba-long-coherence .. eqn:Dee-long-coherence):")
assert_eq("B^a", B[a], targets.ADIABATIC["B^a"])
assert_eq("B^e", B[e], targets.ADIABATIC["B^e"])
assert_eq("D^aa", D[(a, a)], targets.ADIABATIC["D^aa"])
assert_eq("D^ae", D[(a, e)], targets.ADIABATIC["D^ae"])
assert_eq("D^ee", D[(e, e)], targets.ADIABATIC["D^ee"])
assert_eq("B^M0", B[M0], targets.ADIABATIC["B^M0"])

print("Block structure: (a,e) decouples; M0 couples only to omega:")
offblock = [D[(a, v)] for v in (Om, inc, om, M0)] + [D[(e, v)] for v in (Om, inc, om, M0)]
assert_eq("D^{a,rest} = D^{e,rest} = 0", sum(sp.Abs(sp.simplify(x)) for x in offblock), 0)
assert_eq("D^{Omega M0}", D[(Om, M0)], targets.ADIABATIC_EULER["D^OmegaM0"])
assert_eq("D^{i M0}", D[(inc, M0)], targets.ADIABATIC_EULER["D^iM0"])

# ---------------------------------------------------------------------------
# 4b. (E, J) sector via the Ito transformation (eqn:B-transform, eqn:D-transform)
# ---------------------------------------------------------------------------
print("(E,J) sector via Ito transform:")
EE_expr = -G * m / (2 * a)
JJ_expr = sp.sqrt(G * m * a * (1 - e**2))
B_ae = [B[a], B[e]]
D_ae = sp.Matrix([[D[(a, a)], D[(a, e)]], [D[(a, e)], D[(e, e)]]])
B_EJ, D_EJ = ito_transform([a, e], [EE_expr, JJ_expr], B_ae, D_ae)
assert_eq("B^E", B_EJ[0], targets.ADIABATIC["B^E"])
assert_eq("B^J", B_EJ[1], targets.ADIABATIC["B^J"])
assert_eq("D^EE", D_EJ[0, 0], targets.ADIABATIC["D^EE"])
assert_eq("D^EJ", D_EJ[0, 1], targets.ADIABATIC["D^EJ"])
assert_eq("D^JJ", D_EJ[1, 1], targets.ADIABATIC["D^JJ"])

# ---------------------------------------------------------------------------
# 4c. Euler-angle sector (Appendix B, adiabatic block)
# ---------------------------------------------------------------------------
print("Euler-angle sector (Appendix B, adiabatic):")
assert_eq("B^Omega", B[Om], targets.ADIABATIC_EULER["B^Omega"])
assert_eq("B^i", B[inc], targets.ADIABATIC_EULER["B^i"])
assert_eq("B^omega", B[om], targets.ADIABATIC_EULER["B^omega"])
assert_eq("D^OmegaOmega", D[(Om, Om)], targets.ADIABATIC_EULER["D^OmegaOmega"])
assert_eq("D^Omegai", D[(Om, inc)], targets.ADIABATIC_EULER["D^Omegai"])
assert_eq("D^Omegaomega", D[(Om, om)], targets.ADIABATIC_EULER["D^Omegaomega"])
assert_eq("D^ii", D[(inc, inc)], targets.ADIABATIC_EULER["D^ii"])
assert_eq("D^iomega", D[(inc, om)], targets.ADIABATIC_EULER["D^iomega"])
assert_eq("D^omegaomega", D[(om, om)], targets.ADIABATIC_EULER["D^omegaomega"])
assert_eq("D^omegaM0", D[(om, M0)], targets.ADIABATIC_EULER["D^omegaM0"])
assert_eq("D^M0M0", D[(M0, M0)], targets.ADIABATIC["D^M0M0"])

# ---------------------------------------------------------------------------
# 4d. Body frame (ehat, qhat, Jhat, M0) via eqn:Matrix / eqn:Dhatmuhatmu
# ---------------------------------------------------------------------------
print("Body-frame sector (eqn:Dhatehate-long .. eqn:DM0M0-long):")
B_eul = [B[Om], B[inc], B[om]]
D_eul = sp.Matrix(3, 3, lambda p, q: D[((Om, inc, om)[p], (Om, inc, om)[q])])
D_eulM0 = [D[(Om, M0)], D[(inc, M0)], D[(om, M0)]]
B_hat, D_hat, D_hatM0 = body_frame(B_eul, D_eul, D_eulM0)
assert_eq("B^ehat = 0", B_hat[0], 0)
assert_eq("B^qhat = 0", B_hat[1], 0)
assert_eq("B^Jhat = 0", B_hat[2], 0)
assert_eq("D^ee_hat", D_hat[0, 0], targets.ADIABATIC["D^ee_hat"])
assert_eq("D^qq_hat", D_hat[1, 1], targets.ADIABATIC["D^qq_hat"])
assert_eq("D^JJ_hat", D_hat[2, 2], targets.ADIABATIC["D^JJ_hat"])
assert_eq("D^eq_hat = 0", D_hat[0, 1], 0)
assert_eq("D^eJ_hat = 0", D_hat[0, 2], 0)
assert_eq("D^qJ_hat = 0", D_hat[1, 2], 0)
assert_eq("D^eM0_hat = 0", D_hatM0[0], 0)
assert_eq("D^qM0_hat = 0", D_hatM0[1], 0)
assert_eq("D^JM0_hat", D_hatM0[2], targets.ADIABATIC["D^JM0_hat"])

print(f"\nTotal time: {time.time()-t0:.0f}s")
sys.exit(summary("check_tidal_adiabatic"))
