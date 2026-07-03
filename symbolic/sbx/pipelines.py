"""End-to-end pipelines producing all tidal Fokker--Planck coefficients.

Both pipelines return dictionaries B[mu], D[(mu, nu)] over the six slow
variables (a, e, Omega, i, omega, M0), with the overall correlator
normalization (eqn:TD-tidal)

    Int <T_ij T_kl> dtau = (G m / (a^3 T_d)) * ONE_ijkl / 15

(``kind='traceful'``) or its traceless counterpart (eqn:traceless;
``kind='traceless'``), in which case ONE_ijkl -> ONE_ijkl - (5/3) d_ij d_kl.

The computation is performed at Omega = 0. This is justified rigorously by the
two rotation-invariance lemmas proved in check_tidal_adiabatic.py:
g^mu(Omega) = Rz(Omega) g^mu(0) Rz(Omega)^T exactly, and the correlator
contraction is invariant under any joint rotation (eqn:identity-rotation).
The d/dOmega drift terms are taken symbolically before substituting Omega = 0.

* ``adiabatic_coefficients``  (sec:longcoherence, eqn:Deltaw1/eqn:Deltaww1):
  average first, then contract; drifts differentiate the averaged force.
* ``whitenoise_coefficients`` (sec:shortcoherence, eqn:Deltaw2/eqn:Deltaww2):
  contract first at fixed orbital phase, average at the end; drifts use the
  fixed-phase derivative (eqn:dEdedM0) through sbx.anomalies.d_fixed_phase.
"""

import sympy as sp

from .symbols import a, e, G, m, Td, Om, M0, SLOW_VARS, ANGLE_TO_SYM
from .averaging import orbit_average
from .anomalies import d_fixed_phase
from .gauss import g_tensor
from .correlators import contract
from .frames import R0

PREF = sp.Rational(1, 15) * G * m / (a**3 * Td)


def _angles_to_symbols(mat):
    """Replace sin/cos of the Euler angles by plain symbols (post-derivative)."""
    return mat.applyfunc(lambda x: x.xreplace(ANGLE_TO_SYM))


def _tensors_at_Om0():
    """Unaveraged g-tensors and their d/dOmega, evaluated at Omega = 0."""
    R_full = R0()
    R_zero = R_full.subs(Om, 0)
    g0 = {}
    gOm = {}
    for var in SLOW_VARS:
        g0[var] = g_tensor(var, R=R_zero)
        gOm[var] = sp.diff(g_tensor(var, R=R_full), Om).subs(Om, 0)
    return g0, gOm


def adiabatic_coefficients(kind="traceful"):
    g0, gOm = _tensors_at_Om0()
    gbar = {v: g0[v].applyfunc(lambda x: sp.simplify(orbit_average(x))) for v in SLOW_VARS}
    gbarOm = {v: gOm[v].applyfunc(lambda x: sp.simplify(orbit_average(x))) for v in SLOW_VARS}
    D = {}
    for i_mu, mu in enumerate(SLOW_VARS):
        for nu in SLOW_VARS[i_mu:]:
            D[(mu, nu)] = sp.expand(contract(gbar[mu], gbar[nu], kind)) * PREF
            D[(nu, mu)] = D[(mu, nu)]
    B = {}
    for mu in SLOW_VARS:
        val = sp.S.Zero
        for nu in SLOW_VARS:
            if nu == Om:
                dg = gbarOm[mu]
            else:
                # plain partials of the closed-form averages (the average is
                # a function of the slow variables only; its e-dependence
                # already includes the measure term)
                dg = gbar[mu].applyfunc(lambda x, v=nu: sp.diff(x, v))
            val += contract(dg, gbar[nu], kind)
        B[mu] = sp.expand(val / 2) * PREF
    return B, D, gbar


# ---------------------------------------------------------------------------
# Outer-product representation for the white-noise (contract-then-average) case
#
# Every force tensor is a sum of outer products, g^mu = sum_alpha
# (-r c^mu_alpha) (R v_alpha)(R v_r)^T with v_alpha the PERIFOCAL frame
# vectors. All correlator contractions only involve dot products, and
# (R u).(R v) = u.v, while orientation derivatives enter through the
# antisymmetric generators K_nu = R^T dR/d(nu). Working with perifocal
# vectors plus K-matrices keeps every intermediate expression tiny.
# ---------------------------------------------------------------------------

from .gauss import gauss_coefficients, FRAME_PF
from .anomalies import r_of_E
from .symbols import inc, om


def _generators():
    """K_nu = R^T dR/dnu at Omega = 0, for nu in (Omega, i, omega)."""
    R = R0()
    K = {}
    for nu in (Om, inc, om):
        K[nu] = sp.simplify((R.T * sp.diff(R, nu)).subs(Om, 0))
    return K


def _outer_g(var):
    """g^mu as a list of (coeff, u, v) with g = sum coeff * (R u)(R v)^T."""
    coeffs = gauss_coefficients(var)
    out = []
    for c_alpha, v_alpha in zip(coeffs, FRAME_PF):
        if c_alpha != 0:
            out.append((-r_of_E * c_alpha, v_alpha, FRAME_PF[0]))
    return out


def _outer_dg(var, nu, K, naive=False):
    """Fixed-phase derivative of g^mu in outer-product form."""
    deriv = (lambda x: sp.diff(x, nu)) if (naive or nu not in (e_sym, M0)) \
        else (lambda x: d_fixed_phase(x, nu))
    terms = []
    for c, u, v in _outer_g(var):
        dc = deriv(c)
        if dc != 0:
            terms.append((dc, u, v))
        if nu in K:  # orientation derivative acts on R only
            terms.append((c, K[nu] * u, v))
            terms.append((c, u, K[nu] * v))
        else:        # e / M0 / a derivative acts on the perifocal vectors
            du = u.applyfunc(deriv)
            if any(x != 0 for x in du):
                terms.append((c, du, v))
            dv = v.applyfunc(deriv)
            if any(x != 0 for x in dv):
                terms.append((c, u, dv))
    return terms


def _contract_outer(X, Y, kind):
    """Correlator contraction of two outer-product lists (see correlators.py)."""
    tr_coeff = sp.Integer(1) if kind == "traceful" else sp.Rational(-2, 3)
    pieces = []
    for c1, u1, v1 in X:
        for c2, u2, v2 in Y:
            t = (tr_coeff * u1.dot(v1) * u2.dot(v2)
                 + u1.dot(u2) * v1.dot(v2)
                 + u1.dot(v2) * v1.dot(u2))
            pieces.append(sp.expand(c1 * c2 * t))
    return sp.Add(*pieces)


from .symbols import e as e_sym


def whitenoise_coefficients(kind="traceful", naive_drift=False):
    """Contract-then-average (eqn:Deltaw2 / eqn:Deltaww2).

    ``naive_drift=True`` deliberately OMITS the fixed-phase dE/de and dE/dM0
    terms (eqn:dEdedM0) -- used by the guard test that proves those terms are
    essential.
    """
    K = _generators()
    G_out = {v: _outer_g(v) for v in SLOW_VARS}

    def avg(scal):
        return orbit_average(scal.xreplace(ANGLE_TO_SYM))

    D = {}
    for i_mu, mu in enumerate(SLOW_VARS):
        for nu in SLOW_VARS[i_mu:]:
            D[(mu, nu)] = avg(_contract_outer(G_out[mu], G_out[nu], kind)) * PREF
            D[(nu, mu)] = D[(mu, nu)]
    B = {}
    for mu in SLOW_VARS:
        val = sp.S.Zero
        for nu in SLOW_VARS:
            dG = _outer_dg(mu, nu, K, naive=naive_drift)
            val += avg(_contract_outer(dG, G_out[nu], kind))
        B[mu] = sp.expand(val / 2) * PREF
    return B, D
