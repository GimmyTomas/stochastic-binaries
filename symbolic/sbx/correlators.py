"""Isotropic correlator tensors and contraction helpers.

Tidal case (eqn:TD-tidal): the integrated correlator is

    Int <T_ij T_kl> dtau = (G m / (a^3 T_d)) * ONE_ijkl / 15 ,
    ONE_ijkl = delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk .

For a stochastic gravitational-wave background the trace is absent
(eqn:traceless): ONE_ijkl -> ONE_ijkl - (5/3) delta_ij delta_kl.

For two 3x3 matrices X, Y the contractions read

    X_ij Y_kl ONE_ijkl        = tr(X) tr(Y) + tr(X Y^T) + tr(X Y)
    X_ij Y_kl (traceless)     = tr(X Y^T) + tr(X Y) - (2/3) tr(X) tr(Y) .

Impulsive case: Q^{alpha beta} = diag(Q_r, Q_t, Q_t) in the orbital frame
(r, phi, z), so  G^mu_alpha G^nu_beta Q^{alpha beta}
= c^mu_r c^nu_r Q_r + (c^mu_phi c^nu_phi + c^mu_z c^nu_z) Q_t.
"""

import sympy as sp


def contract(X, Y, kind="traceful"):
    """X_ij Y_kl contracted with the isotropic 4-tensor (see module docstring)."""
    trX = X.trace()
    trY = Y.trace()
    base = sp.expand(trX * trY) + sp.expand((X * Y.T).trace()) + sp.expand((X * Y).trace())
    if kind == "traceful":
        return base
    if kind == "traceless":
        return base - sp.Rational(5, 3) * sp.expand(trX * trY)
    raise ValueError(kind)


def q_contract(row_mu, row_nu, Qr, Qt):
    """G^mu_alpha G^nu_beta Q^{alpha beta} for Q = diag(Qr, Qt, Qt)."""
    cr1, cp1, cz1 = row_mu
    cr2, cp2, cz2 = row_nu
    return cr1 * cr2 * Qr + (cp1 * cp2 + cz1 * cz2) * Qt
