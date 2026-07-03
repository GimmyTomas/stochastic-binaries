"""Unit tests for sbx.averaging: every moment table entry is compared against
50-digit mpmath quadrature at several eccentricities."""

import pathlib
import sys
import time

import mpmath as mp
import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import a, e, E, U, LNU
from sbx import averaging as av

mp.mp.dps = 50
ECCS = [mp.mpf("0.1"), mp.mpf("0.37"), mp.mpf("0.9")]
FAILURES = []


def num_avg(fexpr, ecc, jacobian=False):
    """(1/2pi) Int fexpr(E) [ * (1-e cosE) ] dE with mpmath."""
    f = sp.lambdify((e, E), av.to_explicit(fexpr), "mpmath")
    if jacobian:
        g = lambda x: f(ecc, x) * (1 - ecc * mp.cos(x))
    else:
        g = lambda x: f(ecc, x)
    return mp.quad(g, [0, mp.pi, 2 * mp.pi]) / (2 * mp.pi)


def check(name, sym_val, fexpr, jacobian=False, tol=1e-40):
    fnum = sp.lambdify(e, sym_val, "mpmath")
    worst = mp.mpf(0)
    for ecc in ECCS:
        diff = abs(fnum(ecc) - num_avg(fexpr, ecc, jacobian))
        worst = max(worst, diff)
    ok = worst < tol
    print(f"  {name:34s} worst |diff| = {mp.nstr(worst, 3):12s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILURES.append(name)


t0 = time.time()
print("Wallis moments <cos^m E>:")
for mm in range(0, 7):
    check(f"<cos^{mm}>", av.wallis(mm), sp.cos(E) ** mm)

print("Positive-power moments <U^q>:")
for q in range(0, 7):
    check(f"<U^{q}>", av.avg_U(q), U**q)

print("Negative-power moments <U^-p>:")
for p in range(1, 7):
    check(f"<U^-{p}>", av.avg_U_inv(p), U ** (-p))

print("Log moments <U^q log U>:")
for q in range(-4, 5):
    check(f"<U^{q} logU>", av.avg_U_log(q), U**q * LNU)

print("orbit_average on composite expressions (with Jacobian):")
r = a * U
SQ = sp.sqrt(1 - e**2)
vr = e * sp.sin(E) / U
vphi = SQ / U
composites = {
    "<r^2>": (r**2, a**2 * (2 + 3 * e**2) / 2),
    "<r^4>": (r**4, a**4 * (8 + 40 * e**2 + 15 * e**4) / 8),
    "<3r~^2vr^2+r~^2vphi^2>": (3 * U**2 * vr**2 + U**2 * vphi**2, (2 + e**2) / 2),
    "<vr^2 LNU>": (vr**2 * LNU, None),
    "<cos^3 sin^2 /U^2 * LNU>": (sp.cos(E) ** 3 * sp.sin(E) ** 2 / U**2 * LNU, None),
}
for name, (fexpr, expected) in composites.items():
    val = av.orbit_average(fexpr)
    subs_a = val.subs(a, 1) if val.has(a) else val
    fnum = sp.lambdify(e, subs_a, "mpmath")
    worst = mp.mpf(0)
    for ecc in ECCS:
        target = num_avg(fexpr.subs(a, 1), ecc, jacobian=True)
        worst = max(worst, abs(fnum(ecc) - target))
    ok = worst < mp.mpf("1e-38")
    print(f"  {name:34s} worst |diff| = {mp.nstr(worst, 3):12s} {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILURES.append(name)
    if expected is not None:
        if sp.simplify(val - expected) != 0:
            print(f"    ^ closed form mismatch: {sp.simplify(val)} != {expected}")
            FAILURES.append(name + " (closed form)")

print(f"\n{len(FAILURES)} failures; elapsed {time.time()-t0:.1f}s")
sys.exit(1 if FAILURES else 0)
