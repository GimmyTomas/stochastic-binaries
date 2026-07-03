"""Gauss's planetary equations and the force tensors built from them.

The perturbing acceleration is decomposed on the orbital frame
(rhat, phihat, zhat), and Gauss's equations (eqn:dot-a .. eqn:dot-M0) give

    dot w^mu = sum_alpha  c^mu_alpha(w, E) * a^alpha ,   alpha in (r, phi, z).

``gauss_coefficients(var)`` returns the row (c_r, c_phi, c_z) in U-form.
These same coefficients are the matrix G^mu_alpha = d(dot w^mu)/d(a^alpha)
used for impulsive encounters (eqn:F-impulsive).

For tidal perturbations the acceleration is a_i = -T_ij r_j, so that
(eqn:Fs-tidal)  dot w^mu = g^mu_ij T_ij  with

    g^mu_ij = -r * sum_alpha c^mu_alpha * rhat_i * (ehat_alpha)_j ,

where all direction vectors can be expressed either in the perifocal frame
(R = identity) or in the fixed space frame (R = R0(Omega, i, omega)).
Only the part of g symmetric in (ij) contributes when contracting with the
symmetric tidal tensor; the isotropic correlator contraction in
:mod:`sbx.correlators` symmetrizes automatically.
"""

import sympy as sp

from .symbols import a, e, E, G, m, U, Om, inc, om, M0, SQ
from .anomalies import cosphi, sinphi, one_plus_ecosphi, r_of_E

# convenient shorthands (U-form)
_sinwphi = sp.sin(om) * cosphi + sp.cos(om) * sinphi     # sin(omega + phi)
_coswphi = sp.cos(om) * cosphi - sp.sin(om) * sinphi     # cos(omega + phi)


def gauss_coefficients(var):
    """Row (c_r, c_phi, c_z) of Gauss's equations for the slow variable ``var``.

    Transcribed from eqn:dot-a .. eqn:dot-M0 of the paper, with
    cos(phi), sin(phi), r, and 1 + e*cos(phi) in U-form.
    """
    r = r_of_E
    if var == a:
        pref = 2 * a ** sp.Rational(3, 2) / sp.sqrt(G * m * (1 - e**2))
        return (pref * e * sinphi, pref * one_plus_ecosphi, sp.Integer(0))
    if var == e:
        pref = sp.sqrt(a * (1 - e**2) / (G * m))
        return (pref * sinphi, pref * (cosphi + sp.cos(E)), sp.Integer(0))
    if var == Om:
        return (sp.Integer(0), sp.Integer(0),
                _sinwphi * r / (sp.sqrt(G * m * a * (1 - e**2)) * sp.sin(inc)))
    if var == inc:
        return (sp.Integer(0), sp.Integer(0),
                _coswphi * r / sp.sqrt(G * m * a * (1 - e**2)))
    if var == om:
        pref = sp.sqrt(a * (1 - e**2) / (G * m * e**2))
        # (1 + e cos phi) = (1 - e^2)/U keeps all denominators as powers of U
        return (-pref * cosphi,
                pref * (2 + e * cosphi) / one_plus_ecosphi * sinphi,
                -sp.cos(inc) * _sinwphi * r
                / (sp.sqrt(G * m * a * (1 - e**2)) * sp.sin(inc)))
    if var == M0:
        pref = sp.sqrt(a / (G * m)) * (1 - e**2) / e
        return (pref * (cosphi - 2 * e * U / (1 - e**2)),
                -pref * (1 + U / (1 - e**2)) * sinphi,
                sp.Integer(0))
    raise ValueError(f"unknown slow variable {var}")


# Orbital-frame unit vectors in the perifocal basis (columns of R0 act on these)
RHAT_PF = sp.Matrix([cosphi, sinphi, 0])
PHIHAT_PF = sp.Matrix([-sinphi, cosphi, 0])
ZHAT_PF = sp.Matrix([0, 0, 1])
FRAME_PF = (RHAT_PF, PHIHAT_PF, ZHAT_PF)


def g_tensor(var, R=None):
    """3x3 tidal force tensor g^mu_ij (eqn:Fs-tidal), U-form.

    ``R`` rotates the perifocal basis into the space frame; ``R=None`` uses the
    identity (perifocal components), which is sufficient for coefficients that
    do not require derivatives of the orientation, thanks to the rotation
    invariance of the isotropic correlator (eqn:identity-rotation).
    """
    coeffs = gauss_coefficients(var)
    vecs = FRAME_PF if R is None else tuple(R * v for v in FRAME_PF)
    rhat = vecs[0]
    gmat = sp.zeros(3, 3)
    # a^alpha = ehat_alpha . a with a_i = -T_ij r_j, so the acceleration
    # direction contracts the FIRST index of T:  g_ij = -r c_alpha e^alpha_i rhat_j
    for c_alpha, ehat_alpha in zip(coeffs, vecs):
        if c_alpha == 0:
            continue
        gmat += -r_of_E * c_alpha * (ehat_alpha * rhat.T)
    return gmat
