"""Traceless (gravitational-wave background) tidal correlator (sec:blas).

A stochastic GW background acts like a tidal field whose correlator has the
trace removed (eqn:traceless):

    ONE_ijkl -> ONE_ijkl - (5/3) delta_ij delta_kl .

This script re-runs both tidal pipelines (adiabatic and white-noise) with the
traceless correlator and verifies:

* the coefficients LISTED in the draft as modified
  (eqn:DhatJhatJ-traceless .. eqn:DM0M0-traceless);
* the draft's claim that ALL OTHER coefficients are unchanged;
* that the drift still vanishes in the body frame;
* and it prints the differences B_wn - B_ad, D_wn - D_ad, which are the
  predicted Fokker--Planck coefficients for a GW background with
  Omega_gw ~ 1/f (eqn:B-check-blas, eqn:D-check-blas).
"""

import pathlib
import sys
import time

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import a, e, G, m, Td, Om, inc, om, M0
from sbx.pipelines import adiabatic_coefficients, whitenoise_coefficients
from sbx.frames import body_frame
from sbx.ito import ito_transform
from sbx.verify import assert_eq, summary, canon
from sbx import targets

t0 = time.time()


def to_body(B, D):
    B_eul = [B[Om], B[inc], B[om]]
    D_eul = sp.Matrix(3, 3, lambda p, q: D[((Om, inc, om)[p], (Om, inc, om)[q])])
    D_eulM0 = [D[(Om, M0)], D[(inc, M0)], D[(om, M0)]]
    return body_frame(B_eul, D_eul, D_eulM0)


print("Adiabatic pipeline with traceless correlator...")
B_ad, D_ad, _ = adiabatic_coefficients("traceless")
print(f"  done ({time.time()-t0:.0f}s)")
B_hat, D_hat, D_hatM0 = to_body(B_ad, D_ad)

print("Adiabatic, changed coefficients (eqn:DhatJhatJ-traceless block):")
assert_eq("D^JJ_hat", D_hat[2, 2], targets.TRACELESS_ADIABATIC["D^JJ_hat"])
assert_eq("D^JM0_hat", D_hatM0[2], targets.TRACELESS_ADIABATIC["D^JM0_hat"])
assert_eq("D^M0M0", D_ad[(M0, M0)], targets.TRACELESS_ADIABATIC["D^M0M0"])

print("Adiabatic, coefficients that must be UNCHANGED:")
assert_eq("B^a", B_ad[a], targets.ADIABATIC["B^a"])
assert_eq("B^e", B_ad[e], targets.ADIABATIC["B^e"])
assert_eq("D^aa", D_ad[(a, a)], targets.ADIABATIC["D^aa"])
assert_eq("D^ae", D_ad[(a, e)], targets.ADIABATIC["D^ae"])
assert_eq("D^ee", D_ad[(e, e)], targets.ADIABATIC["D^ee"])
assert_eq("D^ee_hat", D_hat[0, 0], targets.ADIABATIC["D^ee_hat"])
assert_eq("D^qq_hat", D_hat[1, 1], targets.ADIABATIC["D^qq_hat"])
assert_eq("B^ehat = B^qhat = B^Jhat = 0",
          sp.Abs(sp.simplify(B_hat[0])) + sp.Abs(sp.simplify(B_hat[1])) + sp.Abs(sp.simplify(B_hat[2])), 0)
assert_eq("B^M0 = 0", B_ad[M0], 0)

print("White-noise pipeline with traceless correlator...")
t1 = time.time()
B_wn, D_wn = whitenoise_coefficients("traceless")
print(f"  done ({time.time()-t1:.0f}s)")
Bw_hat, Dw_hat, Dw_hatM0 = to_body(B_wn, D_wn)

print("White-noise, changed coefficients (traceless block):")
assert_eq("B^a", B_wn[a], targets.TRACELESS_WHITENOISE["B^a"])
assert_eq("B^e", B_wn[e], targets.TRACELESS_WHITENOISE["B^e"])
assert_eq("D^aa", D_wn[(a, a)], targets.TRACELESS_WHITENOISE["D^aa"])
assert_eq("D^ae", D_wn[(a, e)], targets.TRACELESS_WHITENOISE["D^ae"])
assert_eq("D^ee", D_wn[(e, e)], targets.TRACELESS_WHITENOISE["D^ee"])
assert_eq("D^JJ_hat", Dw_hat[2, 2], targets.TRACELESS_WHITENOISE["D^JJ_hat"])
assert_eq("D^JM0_hat", Dw_hatM0[2], targets.TRACELESS_WHITENOISE["D^JM0_hat"])
assert_eq("D^M0M0", D_wn[(M0, M0)], targets.TRACELESS_WHITENOISE["D^M0M0"])

print("White-noise (E,J) sector via Ito (B^E, D^EE quoted in the draft):")
EE_expr = -G * m / (2 * a)
JJ_expr = sp.sqrt(G * m * a * (1 - e**2))
B_EJ, D_EJ = ito_transform([a, e], [EE_expr, JJ_expr],
                           [B_wn[a], B_wn[e]],
                           sp.Matrix([[D_wn[(a, a)], D_wn[(a, e)]],
                                      [D_wn[(a, e)], D_wn[(e, e)]]]))
assert_eq("B^E", B_EJ[0], targets.TRACELESS_WHITENOISE["B^E"])
assert_eq("D^EE", D_EJ[0, 0], targets.TRACELESS_WHITENOISE["D^EE"])

print("White-noise, coefficients that must be UNCHANGED:")
assert_eq("D^ee_hat", Dw_hat[0, 0], targets.WHITENOISE["D^ee_hat"])
assert_eq("D^qq_hat", Dw_hat[1, 1], targets.WHITENOISE["D^qq_hat"])
assert_eq("B^ehat = B^qhat = B^Jhat = 0",
          sp.Abs(sp.simplify(Bw_hat[0])) + sp.Abs(sp.simplify(Bw_hat[1])) + sp.Abs(sp.simplify(Bw_hat[2])), 0)
assert_eq("B^M0 = 0", B_wn[M0], 0)

# ---------------------------------------------------------------------------
# Informational: the Blas--Jenkins combination (white-noise minus adiabatic)
# for a GW background with Omega_gw ~ 1/f (eqn:B-check-blas, eqn:D-check-blas)
# ---------------------------------------------------------------------------
print("\nPredicted GW-background coefficients (white-noise minus adiabatic,")
print("traceless; eqn:B-check-blas / eqn:D-check-blas), for reference:")
for name, wn, ad in [
    ("B^a", B_wn[a], B_ad[a]),
    ("B^e", B_wn[e], B_ad[e]),
    ("D^aa", D_wn[(a, a)], D_ad[(a, a)]),
    ("D^ae", D_wn[(a, e)], D_ad[(a, e)]),
    ("D^ee", D_wn[(e, e)], D_ad[(e, e)]),
]:
    print(f"  {name}_gw = {sp.simplify(sp.factor(sp.simplify(canon(wn - ad))))}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
sys.exit(summary("check_tidal_traceless_gw"))
