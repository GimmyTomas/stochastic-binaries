"""Impulsive-encounter Fokker--Planck coefficients for arbitrary kick
covariances Q_r(r), Q_t(r).

The phase-space force is F^mu = G^mu_j Dv^j (space frame; eqn:F-impulsive),
with G^mu_j = G^mu_alpha ehat^alpha_j built from Gauss's equations. For an
isotropic bath the kick covariance is Q^{jk} = Q_t d_jk + (Q_r - Q_t)
rhat_j rhat_k (orbital-frame diag(Q_r, Q_t, Q_t)), and

    B^mu   = (1/2) < d_nu(G^mu_j) G^nu_k Q^{jk} >_orb      (eqn:Bmu-impulsive)
    D^munu = < G^mu_j G^nu_k Q^{jk} >_orb                  (eqn:Dmunu-impulsive)

where d_nu is the fixed-fast-phase derivative (eqn:dEdedM0). The kick is
fixed in space, so d_nu acts on the Gauss coefficients AND on the orbital
frame vectors (through the generators K_nu = R^T dR/dnu and through phi(E,e)),
but NOT on the kick covariances.

``impulsive_coefficients(Qr, Qt)`` accepts either the opaque symbols
(sbx.symbols.QRS/QTS; results come out in moment-symbol form) or explicit
U-form expressions such as the point-mass kappa*(L_a + 2*LNU - 2/3).
"""

import sympy as sp

from .symbols import (a, e, G, m, E, U, Om, M0, SLOW_VARS, QRS, QTS, SQ,
                      ANGLE_TO_SYM)
from .averaging import orbit_average
from .anomalies import d_fixed_phase, r_of_E
from .gauss import gauss_coefficients, FRAME_PF
from .pipelines import _generators

KGEN = _generators()

# Extra "rows" for the direct energy / angular-momentum observables:
# Edot = v . Dv  =>  G^E = v (velocity vector);  Jdot = r a^phi  =>  G^J = r phihat.
G_ROW_E = (sp.sqrt(G * m / a) * e * sp.sin(E) / U,
           sp.sqrt(G * m / a) * SQ / U,
           sp.Integer(0))
G_ROW_J = (sp.Integer(0), r_of_E, sp.Integer(0))


def vec_G(rows):
    """Force vector as a list of (coeff, perifocal_vector) pairs."""
    return [(c, v) for c, v in zip(rows, FRAME_PF) if c != 0]


def vec_dG(rows, nu):
    """Fixed-phase derivative of the space-frame force vector (perifocal rep.)."""
    terms = []
    for c, v in vec_G(rows):
        dc = d_fixed_phase(c, nu)
        if dc != 0:
            terms.append((dc, v))
        if nu in KGEN:
            terms.append((c, KGEN[nu] * v))
        else:
            dv = v.applyfunc(lambda x: d_fixed_phase(x, nu))
            if any(x != 0 for x in dv):
                terms.append((c, dv))
    return terms


def q_bilinear_vec(X, Y, Qr=QRS, Qt=QTS):
    """x_j y_k Q^{jk} with Q^{jk} = Q_t d_jk + (Q_r - Q_t) rhat_j rhat_k."""
    v_r = FRAME_PF[0]
    pieces = []
    for c1, u1 in X:
        for c2, u2 in Y:
            pieces.append(sp.expand(c1 * c2 * (
                Qt * u1.dot(u2) + (Qr - Qt) * u1.dot(v_r) * u2.dot(v_r))))
    return sp.Add(*pieces)


def _oa(expr):
    return orbit_average(sp.expand(expr).xreplace(ANGLE_TO_SYM))


def impulsive_coefficients(Qr=QRS, Qt=QTS, with_EJ=True):
    """All impulsive B^mu, D^{mu nu} (slow variables, plus optionally E and J).

    Returns (B, D) dictionaries. Keys are the slow-variable symbols, plus the
    strings "E" and "J" when ``with_EJ``.
    """
    rows = {var: gauss_coefficients(var) for var in SLOW_VARS}
    if with_EJ:
        rows["E"] = G_ROW_E
        rows["J"] = G_ROW_J
    keys = list(rows)
    D = {}
    for i_mu, mu in enumerate(keys):
        for nu in keys[i_mu:]:
            D[(mu, nu)] = _oa(q_bilinear_vec(vec_G(rows[mu]), vec_G(rows[nu]),
                                             Qr, Qt))
            D[(nu, mu)] = D[(mu, nu)]
    B = {}
    for mu in keys:
        val = sp.S.Zero
        for nu in SLOW_VARS:  # the nu-sum runs over the slow variables only
            val += _oa(q_bilinear_vec(vec_dG(rows[mu], nu), vec_G(rows[nu]),
                                      Qr, Qt))
        B[mu] = sp.expand(val / 2)
    return B, D
