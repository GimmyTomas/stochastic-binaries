"""Drift and diffusion coefficients for the figure solvers.

Everything is generated from the paper's (a, e) coefficients by the Ito
change of variables (eqn:B-transform, eqn:D-transform), evaluated numerically
on the grid -- so each entry is directly traceable to a labeled equation of
the draft (and to symbolic/check_* scripts that verify those equations).

Solver variables:
* white-noise tidal  (fig 3):  u = x = ln(a/a0),  v = eps = e^2,
  with T_d(a) = (a0/a)^3 T_d0 (eqn:Td-tidal), units a0 = T_d0 = 1;
* point-mass impulsive (fig 4): u = a/a0 (linear), v = eps = e^2, with the
  FULL effective Coulomb logarithm logLambda(a, e) = logLam0 + 2 log(a/a0)
  + 2 log((1+s)/2) (eqn:logLambda; the figure uses logLam0 = 15) and
  1/T_d*(a) ~ a (eqn:Td*); time in units of T_d0 = T_d(a0, e0=0.5) from
  eqn:Td* with the full logLambda(a0, e0) in the bracket -- as in the
  caption of fig 4;
* generic-Q impulsive (fig 6): u = x = ln(a/a0), v = eps, coefficients from
  E-grid quadrature of the general integrands eqn:Ba-impulsive ..
  eqn:Dee-impulsive given tabulated Q_r(r), Q_t(r).
"""

import numpy as np


# ---------------------------------------------------------------------------
# (a, e) -> (u, eps) Ito transform, generic
# ---------------------------------------------------------------------------
def _ito_ae(Ba, Be, Daa, Dae, Dee, a, e, log_a=True):
    """Transform (a, e) coefficients to (x=ln a or a, eps=e^2)."""
    if log_a:
        Bu = Ba / a - Daa / (2 * a**2)
        Duu = Daa / a**2
        Duv = 2 * e * Dae / a
    else:
        Bu = Ba
        Duu = Daa
        Duv = 2 * e * Dae
    Bv = 2 * e * Be + Dee
    Dvv = 4 * e**2 * Dee
    return Bu, Bv, Duu, Duv, Dvv


# ---------------------------------------------------------------------------
# White-noise tidal (fig 3): eqn:Ba-white-noise .. eqn:Dee-white-noise
# ---------------------------------------------------------------------------
def whitenoise_ae(a, e):
    """Draft (a,e) coefficients with 1/T_d = a^3 (units a0 = T_d0 = 1)."""
    invTd = a**3
    Ba = a * (18 + 19 * e**2) / 30 * invTd
    Be = (28 - 51 * e**2 - 103 * e**4) / (240 * np.maximum(e, 1e-30)) * invTd
    Daa = 2 * a**2 * (2 + e**2) / 15 * invTd
    Dae = -2 * a * e * (1 - e**2) / 15 * invTd
    Dee = 7 * (1 - e**2) * (4 + 5 * e**2) / 120 * invTd
    return Ba, Be, Daa, Dae, Dee


def whitenoise_xe_coeffs():
    """Coefficient callables for the (x = ln a, eps = e^2) solver."""
    def make(k):
        def f(X, V):
            a = np.exp(X)
            e = np.sqrt(np.maximum(V, 1e-30))
            vals = _ito_ae(*whitenoise_ae(a, e), a, e, log_a=True)
            return vals[k]
        return f
    keys = ["Bu", "Bv", "Duu", "Duv", "Dvv"]
    return {key: make(k) for k, key in enumerate(keys)}


# ---------------------------------------------------------------------------
# Point-mass impulsive (fig 4): eqn:Ba-point-mass .. eqn:Dee-point-mass
# ---------------------------------------------------------------------------
def td_ratio_pointmass(e, logLam):
    """T_d / T_d^* = sqrt(1-e^2)/[logLam - 8/3 + 4 log(2 sqrt(1-e^2)/(1+sqrt(1-e^2)))]
    (eqn:Td*; logLam is the full Coulomb logarithm at the evaluation point)."""
    w = np.sqrt(1 - e**2)
    return w / (logLam - 8 / 3 + 4 * np.log(2 * w / (1 + w)))


def loglam_pointmass(a, e, logLam0):
    """Full effective Coulomb logarithm (eqn:logLambda), split around its
    constant part logLam0 = logLambda(a0, e=0):
    L(a, e) = logLam0 + 2 log(a/a0) + 2 log((1+s)/2), s = sqrt(1-e^2);
    units a0 = 1."""
    s = np.sqrt(np.maximum(1 - e**2, 0.0))
    return logLam0 + 2 * np.log(a) + 2 * np.log(0.5 * (1 + s))


def pointmass_ae(a, e, logLam0, invTds0=1.0):
    """Draft (a,e) point-mass coefficients with the full logLambda(a, e)
    built from its constant part logLam0; 1/T_d*(a) = invTds0 * a
    (eqn:Td*: T_d^* ~ 1/a; units a0 = 1)."""
    w = np.sqrt(np.maximum(1 - e**2, 1e-30))
    pref = invTds0 * a / 15
    L = loglam_pointmass(a, e, logLam0)
    e_s = np.maximum(e, 1e-30)
    Ba = pref * a * (7 * L - 32 / 3 - 6 * w)
    Be = pref * (5 * (1 - 3 * e**2) / (4 * e_s) * L
                 + (33 * e**4 - 29 * e**2 + 2) / (12 * e_s**3)
                 + (34 * e**4 - 3 * e**2 - 1) / (6 * e_s**3) * w)
    Daa = pref * a**2 * (4 * L - 32 / 3)
    Dae = pref * a * (-4 * (1 - e**2) * (1 - w) / e_s)
    Dee = pref * (5 / 2 * (1 - e**2) * L
                  - (1 - e**2) * (e**2 + 2) / (6 * e_s**2)
                  - (1 - e**2) * (16 * e**2 - 1) / (3 * e_s**2) * w)
    return Ba, Be, Daa, Dae, Dee


def pointmass_a_eps_coeffs(logLam0=15.0, e0=0.5):
    """Coefficient callables for the (a, eps = e^2) solver, in time units of
    T_d0 = T_d(a0 = 1, e0) evaluated from eqn:Td* with the full
    logLambda(a0, e0) in the bracket (caption of fig:f-point-mass)."""
    # T_d(a0, e0) = T_d^*(a0) * td_ratio; solver time is t/T_d0, so we need
    # 1/T_d^*(a) = a / T_d^*(a0) = a * td_ratio / T_d0  ->  invTds0 = td_ratio
    invTds0 = td_ratio_pointmass(np.asarray(e0), loglam_pointmass(1.0, np.asarray(e0), logLam0))

    def make(k):
        def f(A, V):
            e = np.sqrt(np.maximum(V, 1e-30))
            vals = _ito_ae(*pointmass_ae(A, e, logLam0, invTds0), A, e, log_a=False)
            return vals[k]
        return f
    keys = ["Bu", "Bv", "Duu", "Duv", "Dvv"]
    return {key: make(k) for k, key in enumerate(keys)}


def pointmass_fe_reference(e_grid, logLam0):
    """a0-slice steady-state references for the fig 4 overlay/data file.

    Returns (h_exact, h_exp, h_th), each normalized to unit integral on e_grid:
    * h_exact -- integrating-factor solution h = exp(int 2 B^e/D^ee de)/D^ee of
      the a0-slice ODE B^e h = (1/2) d_e(D^ee h) with the full logLambda(a0, e)
      (the cross term d_a(D^ae f) is O(1/logLam0^2), see sec:point-mass);
    * h_exp -- its O(1/logLam0) expansion, i.e. eqn:f-ss-point-mass at a = a0;
    * h_th -- thermal 2e (leading log).
    """
    ones = np.ones_like(e_grid)
    _, Bhat, _, _, Dhat = pointmass_ae(ones, e_grid, logLam0)  # a = a0 = 1
    integrand = 2 * Bhat / Dhat
    u = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                         * np.diff(e_grid))])
    h_exact = np.clip(np.exp(u - u.max()) / Dhat, 0.0, None)
    w = np.sqrt(np.maximum(1 - e_grid**2, 0.0))
    h_exp = e_grid * (1 + (4 * w - 4 * np.log(1 + w) + 2 * np.log(2)) / logLam0)
    h_th = 2 * e_grid
    return tuple(h / np.trapz(h, e_grid) for h in (h_exact, h_exp, h_th))


# ---------------------------------------------------------------------------
# Generic-Q impulsive (fig 6): eqn:Ba-impulsive .. eqn:Dee-impulsive
# ---------------------------------------------------------------------------
def generic_q_ae(a_grid, e_grid, Qr_of_r, Qt_of_r, Gm=1.0, NE=192):
    """Orbit-average the general impulsive integrands on an (a, e) grid.

    Returns Ba, Be, Daa, Dae, Dee arrays of shape a_grid.shape (physical
    units; Q's in units consistent with Gm and the r-argument).
    """
    E = np.linspace(0, 2 * np.pi, NE, endpoint=False) + np.pi / NE
    a = a_grid[..., None]
    e = np.maximum(e_grid[..., None], 1e-12)
    U = 1 - e * np.cos(E)
    r = a * U
    Qr = Qr_of_r(r)
    Qt = Qt_of_r(r)
    vr2 = (e * np.sin(E) / U) ** 2
    vp2 = (1 - e**2) / U**2
    mix = vr2 * Qr + vp2 * Qt

    def avg(x):
        return (x * U).mean(axis=-1)

    e1 = np.maximum(e_grid, 1e-12)
    Ba = a_grid**2 / Gm * avg(Qr + 2 * Qt + 4 * mix)
    Be = a_grid / Gm * avg((1 - e**2) / (2 * e) * (Qr + 2 * Qt)
                           - (1 - e**2) ** 2 / (2 * e**3) * mix
                           - (1 + e**2) / (2 * e**3) * U**2 * Qt
                           + (1 - e**4) / e**3 * Qt)
    Daa = 4 * a_grid**3 / Gm * avg(mix)
    Dae = 2 * a_grid**2 * (1 - e1**2) / (Gm * e1) * avg(mix - Qt)
    Dee = a_grid / Gm * avg((1 - e**2) ** 2 / e**2 * mix
                            + (1 - e**2) / e**2 * U**2 * Qt
                            - 2 * (1 - e**2) ** 2 / e**2 * Qt)
    return Ba, Be, Daa, Dae, Dee


def selfcheck_generic_q():
    """Validation gate: Q_r = 3A r^2, Q_t = A r^2 with A = Gm/(15 a^3 Td)
    must reproduce the closed-form white-noise tidal coefficients."""
    a = np.array([1.0])
    for e in (0.31, 0.62, 0.9):
        ev = np.array([e])
        A = 1 / 15  # Gm = Td = a = 1
        Ba, Be, Daa, Dae, Dee = generic_q_ae(
            a, ev, lambda r: 3 * A * r**2, lambda r: A * r**2, NE=4096)
        tgt = whitenoise_ae(np.array([1.0]), ev)
        tgt = [t / 1.0 for t in tgt]  # invTd = a^3 = 1
        worst = max(abs(x - t).max() / max(abs(t).max(), 1e-30)
                    for x, t in zip((Ba, Be, Daa, Dae, Dee), tgt))
        assert worst < 1e-9, f"generic-Q gate failed at e={e}: {worst:.2e}"
    return True


if __name__ == "__main__":
    selfcheck_generic_q()
    print("generic-Q validation gate: PASS (tidal limit reproduces the "
          "closed-form white-noise coefficients to <1e-9)")
