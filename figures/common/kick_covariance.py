"""Kick covariances Q_r(r), Q_t(r) for uniform-density spherical perturbers
(sec:kick-covariance-tensor, Table I row "Homog. sphere").

All lengths scale with the perturber radius R, so the in-plane integrals are
one-parameter functions of u = r_perp/R:

    Ysum(u)  = Y_perp + Y_par = 4 pi Int rho~(k)^2 (1 - J0(k u)) dk/k ,
    Ydiff(u) = Y_perp - Y_par = 4 pi Int rho~(k)^2 J2(k u) dk/k ,
    rho~(k)  = 3 (sin k - k cos k)/k^3          (dimensionless k = k R).

Because rho~(k)^2 decays like k^-4 these integrals converge absolutely and a
period-resolved trapezoidal rule is accurate; the analytic mean tail
4 pi * (9/2)/(4 K^4) is added beyond the truncation K. An independent
REAL-SPACE check (direct 2D quadrature of Int |y|^2 d^2 b with the form
factor I(b) of Table I) gates the table before it is used.

The theta-integrals (eqn:Qr, eqn:Q) then give, for a Maxwellian bath,

    Q_r(r) = C0 Int_0^pi Yperp(r sin th / R) sin^3 th dth ,
    Q(r)   = C0 Int_0^pi Ysum (r sin th / R) sin th  dth ,   Q_t = (Q - Q_r)/2,
    C0     = 2 G^2 m_*^2 n <V^-1> = 2 G^2 m_*^2 n sqrt(2/pi)/sigma .
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import j0, jn, roots_legendre


def rho_tilde(k):
    k = np.asarray(k)
    out = np.ones_like(k)
    small = k < 1e-4
    kb = k[~small]
    out[~small] = 3 * (np.sin(kb) - kb * np.cos(kb)) / kb**3
    out[small] = 1 - k[small] ** 2 / 10
    return out


def _fourier_pair(u, K=60.0):
    """(Ysum, Ydiff) at scaled separation u via the Fourier route."""
    dk = min(2e-3, np.pi / (12 * max(u, 1.0)))
    k = np.arange(dk / 2, K, dk)
    w2 = rho_tilde(k) ** 2
    ysum = 4 * np.pi * np.trapz(w2 * (1 - j0(k * u)) / k, k)
    ydiff = 4 * np.pi * np.trapz(w2 * jn(2, k * u) / k, k)
    ysum += 4 * np.pi * 4.5 / (4 * K**4)  # analytic mean tail of rho~^2 ~ (9/2)/k^4
    return ysum, ydiff


def _I_of_b(b):
    return np.where(b < 1, 1 - np.abs(1 - np.minimum(b, 1.0) ** 2) ** 1.5, 1.0)


def _realspace_pair(u):
    """(Ysum, Ydiff) by direct 2D quadrature (independent gate), R = 1."""
    def y_vec(bx, by):
        b1sq = bx**2 + by**2 + 1e-300
        b2x, b2y = bx - u, by
        b2sq = b2x**2 + b2y**2 + 1e-300
        f1 = _I_of_b(np.sqrt(b1sq)) / b1sq
        f2 = _I_of_b(np.sqrt(b2sq)) / b2sq
        return f2 * b2x - f1 * bx, f2 * b2y - f1 * by

    B = 150.0 * max(1.0, u)  # |y|^2 ~ b^-4 at large b: truncation error ~ B^-3

    def inner(bx, comp):
        val, _ = quad(lambda by: y_vec(bx, by)[comp] ** 2, 0, B,
                      points=[0.5, 1.0, 2.0, u, u + 1], limit=200)
        return 2 * val  # symmetric in by

    pts = sorted(set([-u - 1, -1.0, 0.0, 0.5 * u,
                      u - 1 if u > 1 else 0.4 * u, u, u + 1]))
    yper, _ = quad(lambda bx: inner(bx, 0), -B, B, points=pts, limit=400)
    ypar, _ = quad(lambda bx: inner(bx, 1), -B, B, points=pts, limit=400)
    return yper + ypar, yper - ypar


class SphereKickCovariance:
    """Tabulated Q_r(r), Q_t(r) for uniform-sphere perturbers."""

    def __init__(self, R, G, mstar, n, sigma, n_u=160, gate=True):
        self.R = R
        self.C0 = 2 * G**2 * mstar**2 * n * np.sqrt(2 / np.pi) / sigma
        # exact small-u (tidal) coefficients: 1 - J0(ku) ~ (ku)^2/4 and
        # J2(ku) ~ (ku)^2/8, so Ysum -> pi M2 u^2, Ydiff -> Ysum/2 with
        # M2 = Int rho~(k)^2 k dk
        kk = np.arange(5e-4, 400.0, 1e-3)
        self.M2 = np.trapz(rho_tilde(kk) ** 2 * kk, kk) + 4.5 / (2 * 400.0**2)
        self.u_grid = np.geomspace(0.05, 3e3, n_u)
        pairs = [_fourier_pair(u) for u in self.u_grid]
        self.Ysum = np.array([p[0] for p in pairs])
        self.Ydiff = np.array([p[1] for p in pairs])
        if gate:
            self._gate()
        self._spl_sum = CubicSpline(np.log(self.u_grid), np.log(self.Ysum))
        self._spl_diff = CubicSpline(np.log(self.u_grid), np.log(self.Ydiff))
        # theta quadrature nodes
        xg, wg = roots_legendre(200)
        self.th = 0.5 * np.pi * (xg + 1)
        self.wth = 0.5 * np.pi * wg

    def _gate(self, tol=2e-4):
        worst = 0.0
        for u in (0.3, 1.0, 3.0, 10.0):
            ys_f, yd_f = _fourier_pair(u)
            ys_r, yd_r = _realspace_pair(u)
            worst = max(worst, abs(ys_f / ys_r - 1), abs(yd_f / yd_r - 1))
        if worst > tol:
            raise RuntimeError(f"Fourier/real-space Y gate failed: {worst:.2e}")
        print(f"  kick-covariance gate: Fourier vs real-space rel. diff "
              f"< {worst:.1e}  (PASS)")

    def _ys(self, u):
        """Ysum with exact asymptotic extensions: ~u^2 below the table
        (tidal), Ysum(umax) + 4 pi log(u/umax) above (point-mass log)."""
        u0, u1 = self.u_grid[0], self.u_grid[-1]
        uc = np.clip(u, u0, u1)
        val = np.exp(self._spl_sum(np.log(uc)))
        val = np.where(u < u0, np.pi * self.M2 * u**2, val)
        val = np.where(u > u1, self.Ysum[-1] + 4 * np.pi * np.log(np.maximum(u, u1) / u1), val)
        return val

    def _yd(self, u):
        """Ydiff with exact asymptotics: ~u^2 below (= Ysum/2), 2 pi above."""
        u0, u1 = self.u_grid[0], self.u_grid[-1]
        uc = np.clip(u, u0, u1)
        val = np.exp(self._spl_diff(np.log(uc)))
        val = np.where(u < u0, np.pi * self.M2 * u**2 / 2, val)
        val = np.where(u > u1, 2 * np.pi, val)
        return val

    def Q_rt(self, r):
        """Return (Q_r, Q_t) at separations r (array), physical units."""
        r = np.atleast_1d(np.asarray(r, dtype=float))
        u = r[:, None] * np.sin(self.th)[None, :] / self.R
        ysum = self._ys(u)
        yperp = 0.5 * (ysum + self._yd(u))
        Qr = self.C0 * (yperp * np.sin(self.th) ** 3 * self.wth).sum(axis=1)
        Qtot = self.C0 * (ysum * np.sin(self.th) * self.wth).sum(axis=1)
        return Qr, 0.5 * (Qtot - Qr)

    def tidal_amplitude(self):
        """A = lim_{r->0} Q_t/r^2 (the white-noise tidal limit)."""
        r0 = 1e-3 * self.R
        Qr, Qt = self.Q_rt(np.array([r0]))
        return float(Qt[0] / r0**2)


if __name__ == "__main__":
    # quick standalone gate + tidal-limit sanity (dimensionless constants)
    cov = SphereKickCovariance(R=1.0, G=1.0, mstar=1.0, n=1.0, sigma=1.0)
    A = cov.tidal_amplitude()
    # eqn:Qt-large for a GAUSSIAN profile does not apply here, but the sphere
    # tidal amplitude equals C0 * (theta integrals of the u^2 coefficients);
    # check consistency of Q_r/Q_t -> 3 in the tidal limit instead:
    Qr, Qt = cov.Q_rt(np.array([1e-3]))
    print(f"tidal limit Q_r/Q_t = {Qr[0]/Qt[0]:.6f} (expect 3)")
    Qr, Qt = cov.Q_rt(np.array([1e3]))
    print(f"point-mass regime Q_r/Q_t at r/R=1e3: {Qr[0]/Qt[0]:.4f} (expect ~1)")
