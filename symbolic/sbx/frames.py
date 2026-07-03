"""Rotations and the body-frame (ehat, qhat, Jhat) change of basis.

R0(Omega, i, omega) = Rz(Omega) Rx(i) Rz(omega) rotates the cartesian axes
onto the orbital axes (ehat, qhat, Jhat).

The Euler-angle rates map onto body-frame angular velocities through
(eqn:Matrix)

    M^(ehat,qhat,Jhat)_(Omega,i,omega) = [[sin i sin omega,  cos omega, 0],
                                          [sin i cos omega, -sin omega, 0],
                                          [cos i,            0,         1]] ,

and the drift/diffusion coefficients transform as (eqn:Dhatmuhatmu)

    B^hat = M B + (1/2) d(M)/d(w^mu) D^{mu nu}   (Ito term),
    D^hat = M D M^T,          D^{hat, M0} = M D^{., M0} .
"""

import sympy as sp

from .symbols import Om, inc, om


def Rz(th):
    return sp.Matrix([[sp.cos(th), -sp.sin(th), 0],
                      [sp.sin(th), sp.cos(th), 0],
                      [0, 0, 1]])


def Rx(th):
    return sp.Matrix([[1, 0, 0],
                      [0, sp.cos(th), -sp.sin(th)],
                      [0, sp.sin(th), sp.cos(th)]])


def R0(Omega=Om, i=inc, omega=om):
    return Rz(Omega) * Rx(i) * Rz(omega)


def Mmat(i=inc, omega=om):
    """Change-of-basis matrix of eqn:Matrix; rows (ehat,qhat,Jhat), cols (Omega,i,omega)."""
    return sp.Matrix([[sp.sin(i) * sp.sin(omega), sp.cos(omega), 0],
                      [sp.sin(i) * sp.cos(omega), -sp.sin(omega), 0],
                      [sp.cos(i), 0, 1]])


EULER_VARS = (Om, inc, om)


def body_frame(B_euler, D_euler, D_euler_M0=None):
    """Transform Euler-angle-sector coefficients to the body frame.

    Parameters
    ----------
    B_euler : 3-vector (B^Omega, B^i, B^omega)
    D_euler : symmetric 3x3 matrix D^{mu nu}, mu,nu in (Omega, i, omega)
    D_euler_M0 : optional 3-vector (D^{Omega M0}, D^{i M0}, D^{omega M0})

    Returns (B_hat, D_hat[, D_hat_M0]) with rows ordered (ehat, qhat, Jhat).
    """
    M = Mmat()
    B_hat = M * sp.Matrix(B_euler)
    # Ito term: (1/2) * dM^{hat}_nu/d(w^mu) * D^{mu nu}
    for hat in range(3):
        corr = sp.S.Zero
        for mu_idx, mu_var in enumerate(EULER_VARS):
            for nu_idx in range(3):
                corr += sp.diff(M[hat, nu_idx], mu_var) * D_euler[mu_idx, nu_idx]
        B_hat[hat] += corr / 2
    D_hat = M * D_euler * M.T
    if D_euler_M0 is None:
        return B_hat, D_hat
    D_hat_M0 = M * sp.Matrix(D_euler_M0)
    return B_hat, D_hat, D_hat_M0
