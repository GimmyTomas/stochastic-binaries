"""Figure 5 (fig:Td(R)): diffusion time T_d (eqn:TD-impulsive), normalized by
T_d^* (eqn:Td*), for impulsive encounters with GAUSSIAN perturbers of size R,
as a function of R/a and for eccentricities e = 0 and 0.9.

Pipeline (sec:kick-covariance-tensor with rho~(k) = exp(-k^2 R^2/2)):
the in-plane integrals eqn:Yperp+Yparallel / eqn:Yperp-Yparallel have closed
forms (no oscillatory quadrature needed; verified in
symbolic/check_kick_covariance.py):

    Y_perp + Y_par = 2 pi Ein(u),  u = r_perp^2/(4 R^2),
                     Ein(u) = log u + gamma_E + E1(u)
    Y_perp - Y_par = 2 pi [1 - (1 - e^{-u})/u] ,

then the theta-integrals eqn:Qr / eqn:Q give Q_r(r), Q_t(r), and the orbit
average <Q_t/r^2> gives T_d. Overlaid: the tidal limit T_d -> 10 (R/a)^2 T_d^*
(eqn:Td-large) and the small-R closed form eqn:Td* with the Gaussian Coulomb
logarithm. The script asserts the draft's claim that the tidal limit is
accurate to better than 9% for all R/a > 1.

Runtime: seconds.
"""

import argparse
import pathlib
import sys

import numpy as np
from scipy.special import exp1, roots_legendre

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.style import setup, COLORS

OUT = pathlib.Path(__file__).resolve().parent / "output"
GAMMA_E = 0.5772156649015329


def ein(u):
    """Ein(u) = log(u) + gamma_E + E1(u), series-protected at small u."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    small = u < 1e-6
    out[small] = u[small] - u[small] ** 2 / 4
    ub = u[~small]
    out[~small] = np.log(ub) + GAMMA_E + exp1(ub)
    return out


def one_minus_ratio(u):
    """1 - (1 - exp(-u))/u, series-protected at small u."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    small = u < 1e-6
    out[small] = u[small] / 2 - u[small] ** 2 / 6
    ub = u[~small]
    out[~small] = 1 - (1 - np.exp(-ub)) / ub
    return out


def td_over_tdstar(R_over_a, e, NE=2000, Nth=400):
    """T_d/T_d^* = (4 pi/3) / (a^2 <I_t(r)/r^2>), units a = 1.

    I_t(r) = (1/2) Int [Ysum/(2pi) sin(th) - Yperp/(2pi) sin^3(th)] dth with
    the closed-form Gaussian Ysum, Ydiff (the 2 pi's cancel against the 4pi/3
    prefactor bookkeeping below; we carry them explicitly).
    """
    xg, wg = roots_legendre(Nth)
    th = 0.5 * np.pi * (xg + 1)
    wth = 0.5 * np.pi * wg
    E = np.linspace(0, 2 * np.pi, NE, endpoint=False) + np.pi / NE
    U = 1 - e * np.cos(E)
    r = U  # a = 1
    rp = r[:, None] * np.sin(th)[None, :]      # (NE, Nth)
    u = rp**2 / (4 * R_over_a**2)
    Ysum = 2 * np.pi * ein(u)
    Ydiff = 2 * np.pi * one_minus_ratio(u)
    Yperp = 0.5 * (Ysum + Ydiff)
    # theta-integrals (eqn:Qr, eqn:Q): I_r = Int Yperp sin^3, I_q = Int Ysum sin
    I_r = (Yperp * np.sin(th)[None, :] ** 3 * wth[None, :]).sum(axis=1)
    I_q = (Ysum * np.sin(th)[None, :] * wth[None, :]).sum(axis=1)
    I_t = 0.5 * (I_q - I_r)
    avg = ((I_t / r**2) * U).mean()            # orbit average with Jacobian
    return (4 * np.pi / 3) / avg


def smallR_asymptote(R_over_a, e):
    """eqn:Td* with the Gaussian logLambda = log(a^2 (1+w)^2/(4R^2)) + gamma_E."""
    w = np.sqrt(1 - e**2)
    logLam = np.log((1 + w) ** 2 / (4 * R_over_a**2)) + GAMMA_E
    return w / (logLam - 8 / 3 + 4 * np.log(2 * w / (1 + w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)

    npts = 40 if args.fast else 110
    Rgrid = np.geomspace(0.01, 30, npts)
    eccs = [0.0, 0.9]
    curves = {e: np.array([td_over_tdstar(R, e) for R in Rgrid]) for e in eccs}

    # draft's claim: tidal limit accurate to <9% for all R/a > 1, any e
    tidal = 10 * Rgrid**2
    worst = 0.0
    for e in eccs + [0.5, 0.99]:
        c = np.array([td_over_tdstar(R, e) for R in Rgrid[Rgrid >= 1]])
        worst = max(worst, np.abs(c / tidal[Rgrid >= 1] - 1).max())
    print(f"max |T_d/(10 R^2/a^2 T_d*) - 1| for R/a >= 1: {worst*100:.1f}%  "
          f"({'PASS' if worst < 0.09 else 'FAIL'} @ 9%)")

    plt = setup()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for i, e in enumerate(eccs):
        ax.loglog(Rgrid, curves[e], color=COLORS[i], label=rf"$e = {e:g}$")
        ax.loglog(Rgrid, smallR_asymptote(Rgrid, e), ":", color=COLORS[i], lw=1)
    ax.loglog(Rgrid, tidal, "k--", lw=1, label=r"$10\,(R/a)^2$ (tidal limit)")
    ax.set_xlim(0.01, 30)
    ax.set_ylim(1e-2, 1e4)
    ax.set_xlabel(r"$R/a$")
    ax.set_ylabel(r"$T_\mathrm{d}/T_\mathrm{d}^*$")
    ax.legend(loc="upper left")
    fig.savefig(outdir / "fig5_Td_vs_R.pdf")
    np.savetxt(outdir / "fig5_data.dat",
               np.column_stack([Rgrid] + [curves[e] for e in eccs]
                               + [smallR_asymptote(Rgrid, e) for e in eccs] + [tidal]),
               header="R_over_a Td_e0 Td_e0p9 asy_e0 asy_e0p9 tidal")
    print(f"wrote {outdir/'fig5_Td_vs_R.pdf'}")


if __name__ == "__main__":
    main()
