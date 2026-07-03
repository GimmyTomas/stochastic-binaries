"""Table I of the paper (tab:I(b)): perturber density profiles rho_*(x),
their form factors I(b) (eqn:imp-formfactor-def) and mass-normalized Fourier
transforms rho~_*(k) (eqn:rho(k)).

For each profile (Plummer, Gaussian, homogeneous sphere, Hernquist) we check:
 (a) mass normalization: Int 4 pi x^2 rho_*(x) dx = m_*;
 (b) I(b) from the definition
       I(b) = Int_-oo^oo M(b sqrt(1+u^2)) / (2 m_* (1+u^2)^{3/2}) du ,
     with M(x) the enclosed mass, against the table entry;
 (c) rho~(k) = (4 pi/(k m_*)) Int x sin(kx) rho_*(x) dx against the table;
 (d) the Hankel-transform relation rho~(k) = k Int_0^oo I(b) J_1(kb) db.
SymPy is used where it succeeds; otherwise the identity is verified with
50-digit mpmath quadrature over a grid of parameters (reported as NUM).
Both the b < R and b > R branches are exercised for the sphere and Hernquist
(via mpmath's analytic continuation of artanh for the latter).
"""

import pathlib
import sys
import time

import mpmath as mp
import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import mstar, Rs
from sbx.verify import assert_eq, summary, RESULTS

mp.mp.dps = 50
t0 = time.time()

x, b, k, u = sp.symbols("x b k u", positive=True)

PROFILES = {
    "Plummer": {
        "rho": 3 * mstar / (4 * sp.pi * Rs**3) * (1 + x**2 / Rs**2) ** sp.Rational(-5, 2),
        "I": b**2 / (b**2 + Rs**2),
        "rhok": k * Rs * sp.besselk(1, k * Rs),
    },
    "Gaussian": {
        "rho": mstar / ((2 * sp.pi) ** sp.Rational(3, 2) * Rs**3) * sp.exp(-x**2 / (2 * Rs**2)),
        "I": 1 - sp.exp(-b**2 / (2 * Rs**2)),
        "rhok": sp.exp(-k**2 * Rs**2 / 2),
    },
    "Homog. sphere": {
        "rho": sp.Piecewise((3 * mstar / (4 * sp.pi * Rs**3), x < Rs), (0, True)),
        "I": sp.Piecewise((1 - (1 - b**2 / Rs**2) ** sp.Rational(3, 2), b < Rs), (1, True)),
        "rhok": 3 * (sp.sin(k * Rs) - k * Rs * sp.cos(k * Rs)) / (k * Rs) ** 3,
    },
    "Hernquist": {
        "rho": mstar * Rs / (2 * sp.pi * x * (x + Rs) ** 3),
        "I": b**2 / (Rs**2 - b**2) * (sp.atanh(sp.sqrt(1 - b**2 / Rs**2))
                                      / sp.sqrt(1 - b**2 / Rs**2) - 1),
        "rhok": 1 + k * Rs * (sp.cos(k * Rs) * (sp.Si(k * Rs) - sp.pi / 2)
                              - sp.sin(k * Rs) * sp.Ci(k * Rs)),
    },
}


def numeric_check(name, f_lhs, f_rhs, samples, tol=1e-30):
    worst = mp.mpf(0)
    for smp in samples:
        try:
            worst = max(worst, abs(f_lhs(*smp) - f_rhs(*smp)))
        except Exception as exc:  # pragma: no cover
            print(f"  [FAIL] {name} (evaluation error: {exc})")
            RESULTS.append((name, "FAIL"))
            return
    ok = worst < tol
    print(f"  [{'NUM ' if ok else 'FAIL'}] {name} (worst |diff| = {mp.nstr(worst, 3)})")
    RESULTS.append((name, "NUM" if ok else "FAIL"))


for pname, data in PROFILES.items():
    print(f"--- {pname} ---")
    rho, I_tab, rhok_tab = data["rho"], data["I"], data["rhok"]

    # (a) mass normalization
    mass = sp.integrate(4 * sp.pi * x**2 * rho, (x, 0, sp.oo))
    assert_eq(f"{pname}: mass normalization", mass, mstar)

    # enclosed mass M(x)
    Mx = sp.integrate(4 * sp.pi * x**2 * rho, (x, 0, x))

    # (b) I(b) from the definition (numeric; the u-integral is hard for sympy)
    Mx_f = sp.lambdify((x, Rs), (Mx / mstar).subs(mstar, 1).rewrite(sp.erf), "mpmath")

    def I_def(bb, RR, Mx_f=Mx_f):
        # split at the kink u* where b sqrt(1+u^2) = R (piecewise profiles)
        pts = [-mp.inf, 0, mp.inf]
        if bb < RR:
            ustar = mp.sqrt((RR / bb) ** 2 - 1)
            pts = [-mp.inf, -ustar, 0, ustar, mp.inf]
        return mp.quad(lambda uu: Mx_f(bb * mp.sqrt(1 + uu**2), RR)
                       / (2 * (1 + uu**2) ** mp.mpf(1.5)), pts)

    I_tab_f = sp.lambdify((b, Rs), I_tab, "mpmath")

    def I_tab_eval(bb, RR):
        if pname == "Hernquist" and bb > RR:
            return mp.re(I_tab_f(mp.mpc(bb), mp.mpc(RR)))
        return I_tab_f(bb, RR)

    samples_b = [(mp.mpf("0.3"), mp.mpf(1)), (mp.mpf("0.9"), mp.mpf(1)),
                 (mp.mpf(2), mp.mpf(1)), (mp.mpf(5), mp.mpf("1.7"))]
    numeric_check(f"{pname}: I(b) from eqn:imp-formfactor-def", I_def,
                  I_tab_eval, samples_b, tol=mp.mpf("1e-40"))

    # (c) rho~(k) from the 3D Fourier transform
    try:
        rhok_def = sp.integrate(4 * sp.pi * x * sp.sin(k * x) * rho / (k * mstar),
                                (x, 0, sp.oo))
        ok = assert_eq(f"{pname}: rho~(k) from Fourier transform",
                       sp.simplify(rhok_def), rhok_tab)
    except Exception:
        ok = False
    if not ok:
        rhok_tab_f = sp.lambdify((k, Rs), rhok_tab, "mpmath")
        rho_f = sp.lambdify((x, Rs), (rho / mstar).subs(mstar, 1), "mpmath")

        def rhok_num(kk, RR):
            return mp.quad(lambda xx: 4 * mp.pi * xx * mp.sin(kk * xx) * rho_f(xx, RR) / kk,
                           [0, RR, 10 * RR, mp.inf])

        numeric_check(f"{pname}: rho~(k) Fourier (numeric)", rhok_num, rhok_tab_f,
                      [(mp.mpf("0.5"), mp.mpf(1)), (mp.mpf(3), mp.mpf(1))],
                      tol=mp.mpf("1e-35"))

    # (d) rho~(k) = k Int I(b) J1(kb) db  (eqn:rho(k)).
    # Write I = 1 + (I - 1): the point-mass part gives k Int J1(kb) db = 1
    # exactly, and (I - 1) decays algebraically (or has compact support), so
    # the remainder converges absolutely; its oscillatory tail is handled by
    # mpmath's quadosc.
    rhok_tab_f = sp.lambdify((k, Rs), rhok_tab, "mpmath")

    def hankel(kk, RR):
        def integrand(bb):
            return kk * (I_tab_eval(bb, RR) - 1) * mp.besselj(1, kk * bb)
        if pname == "Homog. sphere":
            return 1 + mp.quad(integrand, mp.linspace(mp.mpf("1e-12"), RR, 12))
        Bcut = 30 * RR + 30 / kk
        head = mp.quad(integrand, mp.linspace(mp.mpf("1e-12"), Bcut, 60))
        tail = mp.quadosc(integrand, [Bcut, mp.inf], period=2 * mp.pi / kk)
        return 1 + head + tail

    numeric_check(f"{pname}: rho~(k) = k Int I(b) J1(kb) db (eqn:rho(k))",
                  hankel, rhok_tab_f,
                  [(mp.mpf("0.7"), mp.mpf(1)), (mp.mpf(2), mp.mpf(1))],
                  tol=mp.mpf("1e-18"))  # quadosc accuracy floor is ~1e-24

print(f"\nTotal time: {time.time()-t0:.0f}s")
sys.exit(summary("check_form_factors"))
