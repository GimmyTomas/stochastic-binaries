"""Ito change of variables for Fokker--Planck coefficients.

Under w^mu -> wt^mut(w) the coefficients transform as (eqn:B-transform,
eqn:D-transform)

    Bt^mut = (d wt^mut / d w^mu) B^mu
             + (1/2) (d^2 wt^mut / d w^mu d w^nu) D^{mu nu} ,
    Dt^{mut nut} = (d wt/d w)^mut_mu (d wt/d w)^nut_nu D^{mu nu} .
"""

import sympy as sp


def ito_transform(old_vars, new_exprs, B_old, D_old):
    """Transform (B, D) from ``old_vars`` to the variables ``new_exprs(old_vars)``.

    Parameters
    ----------
    old_vars : sequence of n sympy symbols
    new_exprs : sequence of k expressions in ``old_vars``
    B_old : sequence of n drift coefficients
    D_old : n x n symmetric matrix of diffusion coefficients

    Returns (B_new [k], D_new [k x k]).
    """
    n = len(old_vars)
    k = len(new_exprs)
    B_new = []
    for f in new_exprs:
        val = sum(sp.diff(f, old_vars[i]) * B_old[i] for i in range(n))
        val += sp.Rational(1, 2) * sum(
            sp.diff(f, old_vars[i], old_vars[j]) * D_old[i, j]
            for i in range(n) for j in range(n))
        B_new.append(sp.expand(val))
    D_new = sp.zeros(k, k)
    for p in range(k):
        for q in range(k):
            D_new[p, q] = sp.expand(sum(
                sp.diff(new_exprs[p], old_vars[i]) * sp.diff(new_exprs[q], old_vars[j]) * D_old[i, j]
                for i in range(n) for j in range(n)))
    return B_new, D_new
