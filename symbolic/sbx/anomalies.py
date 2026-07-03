"""Anomalies and fixed-phase derivatives.

The true anomaly phi is eliminated in favor of the eccentric anomaly E via

    cos(phi) = (cos E - e)/U,   sin(phi) = sqrt(1-e^2) sin E / U,   U = 1 - e cos E,

and the radius is r = a*U (all expressions in this package are "U-form": they
contain E only through cos(E), sin(E), and the symbols U, LNU = log U).

Fixed-phase derivatives (eqn:dEdedM0). The drift coefficients require partial
derivatives of the force components with respect to the slow variables at
*fixed fast phase* MM, where M = MM + M0 and Kepler's equation reads
MM + M0 = E - e sin E. At fixed MM:

    dE/de  = sin E / U,        dE/dM0 = 1 / U.

Because U and LNU are kept symbolic, the chain rule must include their
dependence on (e, E):  dU/de|_E = -cos E,  dU/dE = e sin E, and similarly for
LNU = log U. This module provides the single chokepoint ``d_fixed_phase``
through which ALL drift computations must go.
"""

import sympy as sp

from .symbols import a, e, E, U, LNU, M0, SQ

cosphi = (sp.cos(E) - e) / U
sinphi = SQ * sp.sin(E) / U
r_of_E = a * U

# 1 + e*cos(phi) = (1 - e^2)/U  -- used to simplify Gauss's equations
one_plus_ecosphi = (1 - e**2) / U


def dE_dvar(var):
    """dE/dvar at fixed fast phase MM (eqn:dEdedM0). Zero unless var in (e, M0)."""
    if var == e:
        return sp.sin(E) / U
    if var == M0:
        return 1 / U
    return sp.Integer(0)


def _d_dE(expr):
    """Total d/dE at fixed slow variables: E explicit + U(E) + LNU(E)."""
    return (sp.diff(expr, E)
            + sp.diff(expr, U) * e * sp.sin(E)
            + sp.diff(expr, LNU) * e * sp.sin(E) / U)


def _d_de_fixed_E(expr):
    """Partial d/de at fixed E: e explicit + U(e) + LNU(e)."""
    return (sp.diff(expr, e)
            + sp.diff(expr, U) * (-sp.cos(E))
            + sp.diff(expr, LNU) * (-sp.cos(E) / U))


def d_fixed_phase(expr, var):
    """d(expr)/d(var) at fixed fast phase MM.

    This is THE chokepoint for all drift computations: it implements
    d/dvar|_MM = d/dvar|_E + (d/dE) * (dE/dvar|_MM), with the U- and
    LNU-chain-rule terms included.
    """
    if var == e:
        return _d_de_fixed_E(expr) + _d_dE(expr) * dE_dvar(e)
    if var == M0:
        return _d_dE(expr) * dE_dvar(M0)
    return sp.diff(expr, var)
