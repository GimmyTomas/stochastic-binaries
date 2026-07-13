"""Shared SymPy symbols for the stochastic-binaries symbolic package.

Conventions follow the paper "Evolution of Binaries Under Stochastic
Perturbations". The slow orbital variables are w = (a, e, Omega, i, omega, M0).

Two special symbols encode structures that make orbit averaging fast:

* ``U``   stands for 1 - e*cos(E)  (E = eccentric anomaly). Expressions are
          built keeping U symbolic, so that every term is a monomial
          cos(E)^c * sin(E)^s * U^q and the average over E reduces to a table
          lookup. Derivatives with respect to (e, E) must use the chain-rule
          helpers in :mod:`sbx.anomalies`, which know that dU/de = -cos(E) and
          dU/dE = e*sin(E).
* ``LNU`` stands for log(U) = log(1 - e*cos(E)). It appears (linearly) in the
          point-mass kick covariances through the Coulomb logarithm.

``w`` (= sqrt(1-e^2)) is used only at *comparison* time by :mod:`sbx.verify`;
during construction and differentiation the explicit sqrt(1-e**2) is used so
that d/de acts correctly.
"""

import sympy as sp

# --- binary / orbit symbols -------------------------------------------------
a, e, G, m = sp.symbols("a e G m", positive=True)
E = sp.Symbol("E", real=True)                 # eccentric anomaly (fast phase)
U = sp.Symbol("U", positive=True)             # U := 1 - e*cos(E)
LNU = sp.Symbol("LNU", real=True)             # LNU := log(U)
w = sp.Symbol("w", positive=True)             # w := sqrt(1 - e^2) (verify-time only)

# Euler angles of the orbit and mean anomaly at epoch
Om = sp.Symbol("Omega", real=True)            # longitude of ascending node
inc = sp.Symbol("i", real=True)               # inclination
om = sp.Symbol("omega", real=True)            # argument of pericenter
M0 = sp.Symbol("M_0", real=True)              # mean anomaly at epoch

# --- environment / normalization symbols ------------------------------------
Td = sp.Symbol("T_d", positive=True)          # diffusion time, eqn:Td-tidal / eqn:Td-impulsive
Tds = sp.Symbol("T_d^*", positive=True)       # T_d^*, eqn:Td*
LogLam = sp.Symbol("logLambda", real=True)    # effective Coulomb logarithm, eqn:logLambda
LogLam0 = sp.Symbol("logLambda_0", positive=True)  # constant part logLambda(a_0, e=0)
a0 = sp.Symbol("a_0", positive=True)          # reference semi-major axis
La = sp.Symbol("L_a", real=True)              # r-independent part of L(r): L(r) = La + 2*LNU
kappa = sp.Symbol("kappa", positive=True)     # 8*sqrt(2*pi)*G^2*m_*^2*n/(3*sigma)
Aamp = sp.Symbol("A", positive=True)          # tidal-limit amplitude: Q_t = A r^2

sigma, mstar, nden, Rs = sp.symbols("sigma m_* n R", positive=True)

# Opaque kick-covariance components Q_r(r), Q_t(r) (functions of r = a*U),
# kept symbolic for the general-perturber impulsive results. Terms must be
# LINEAR in these; sbx.averaging reduces them to opaque moment symbols
# MQr_q / MQt_q = (1/2pi) Int U^q Q_{r,t} dE.
QRS = sp.Symbol("Q_r")
QTS = sp.Symbol("Q_t")

# Frequently used composite quantities
SQ = sp.sqrt(1 - e**2)                        # sqrt(1-e^2), explicit form

# Plain symbols standing for sin/cos of the Euler angles. Heavy computations
# replace the trig *functions* by these symbols (after all angle derivatives
# are taken) so that expansions are pure polynomial operations; the verifier
# canonicalizes both representations to the same symbols.
c_i, s_i = sp.symbols("c_i s_i", real=True)
c_w, s_w = sp.symbols("c_w s_w", real=True)
c_O, s_O = sp.symbols("c_O s_O", real=True)
ANGLE_TO_SYM = {
    sp.cos(inc): c_i, sp.sin(inc): s_i,
    sp.cos(om): c_w, sp.sin(om): s_w,
    sp.cos(Om): c_O, sp.sin(Om): s_O,
}

# Ordered tuple of the six slow variables (paper convention: M0 last)
SLOW_VARS = (a, e, Om, inc, om, M0)
SLOW_NAMES = ("a", "e", "Omega", "i", "omega", "M0")
