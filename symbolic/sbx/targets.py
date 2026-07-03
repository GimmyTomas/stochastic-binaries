"""Every target expression from the paper, transcribed verbatim and keyed by
its LaTeX \\label (or by the section it appears in when unlabeled).

Conventions:
* SQ = sqrt(1-e^2) is kept explicit (the verifier converts it to w);
* EE = -G m/(2 a) is the orbital energy, JJ = sqrt(G m a (1-e^2)) the
  angular momentum magnitude;
* T_d is the diffusion time of eqn:TD-tidal (tidal) / eqn:TD-impulsive
  (impulsive); T_d^* is defined in eqn:Td*; logLambda in eqn:logLambda.
"""

import sympy as sp

from .symbols import a, e, G, m, Td, Tds, LogLam, Om, inc, om, SQ

EE = -G * m / (2 * a)
JJ = sp.sqrt(G * m * a * (1 - e**2))

L = LogLam

# ---------------------------------------------------------------------------
# Adiabatic tidal regime (sec:longcoherence)
# ---------------------------------------------------------------------------
ADIABATIC = {
    # eqn:Ba-long-coherence .. eqn:Dee-long-coherence
    "B^a": sp.Integer(0),
    "B^e": 5 * e * (3 - 5 * e**2) / (24 * Td),
    "D^aa": sp.Integer(0),
    "D^ae": sp.Integer(0),
    "D^ee": 5 * e**2 * (1 - e**2) / (12 * Td),
    # (E, J) block below eqn:Dee-long-coherence
    "B^E": sp.Integer(0),
    "B^J": -5 * JJ * e**2 * (4 - 5 * e**2) / (24 * (1 - e**2) * Td),
    "D^EE": sp.Integer(0),
    "D^EJ": sp.Integer(0),
    "D^JJ": 5 * G * m * a * e**4 / (12 * Td),
    # body frame, eqn:Dhatehate-long .. eqn:DM0M0-long
    "D^ee_hat": (1 + 4 * e**2) ** 2 / (60 * (1 - e**2) * Td),
    "D^qq_hat": (1 - e**2) / (60 * Td),
    "D^JJ_hat": 43 * (1 - e**2) / (60 * Td),
    "D^JM0_hat": -SQ * (67 + 43 * e**2) / (60 * Td),
    "D^M0M0": (123 + 134 * e**2 + 43 * e**4) / (60 * Td),
    "B^M0": sp.Integer(0),
    # orbit-averaged force examples, eqn:gexy block.
    # NOTE on gbar^e: the draft quotes gbar^e_xy = -(5/2) a^{3/2} e sqrt((1-e^2)/(Gm)).
    # The exact calculation (confirmed by direct numerical quadrature of Gauss's
    # equation for edot under an explicit symmetric tidal tensor) gives the
    # nonzero component in the yx slot with a PLUS sign, in the same index
    # convention that makes gbar^a_xy = +sqrt(a^5(1-e^2)/(Gm)) as quoted.
    # This affects only the illustrative example, not any final coefficient.
    "gbar^a_xy": sp.sqrt(a**5 * (1 - e**2) / (G * m)),
    "gbar^e_yx": sp.Rational(5, 2) * a ** sp.Rational(3, 2) * e * sp.sqrt((1 - e**2) / (G * m)),
}

# Appendix B, adiabatic block (sec:BD-orientation)
_D_OmOm_long = (2 + 6 * e**2 + 17 * e**4 - 5 * e**2 * (2 + 3 * e**2) * sp.cos(2 * om)) \
    / (120 * (1 - e**2) * sp.sin(inc) ** 2 * Td)
ADIABATIC_EULER = {
    "B^Omega": -e**2 * (2 + 3 * e**2) * sp.sin(2 * om)
    / (24 * (1 - e**2) * sp.tan(inc) * sp.sin(inc) * Td),
    "B^i": (2 + 6 * e**2 + 17 * e**4 - 5 * e**2 * (2 + 3 * e**2) * sp.cos(2 * om))
    / (240 * (1 - e**2) * sp.tan(inc) * Td),
    "B^omega": e**2 * (2 + 3 * e**2) * (3 + sp.cos(2 * inc)) * sp.sin(2 * om)
    / (96 * (1 - e**2) * sp.sin(inc) ** 2 * Td),
    "D^OmegaOmega": _D_OmOm_long,
    "D^Omegai": e**2 * (2 + 3 * e**2) * sp.sin(2 * om) / (24 * (1 - e**2) * sp.sin(inc) * Td),
    "D^Omegaomega": -_D_OmOm_long * sp.cos(inc),
    "D^ii": (2 + 6 * e**2 + 17 * e**4 + 5 * e**2 * (2 + 3 * e**2) * sp.cos(2 * om))
    / (120 * (1 - e**2) * Td),
    "D^iomega": -e**2 * (2 + 3 * e**2) * sp.sin(2 * om) / (24 * (1 - e**2) * sp.tan(inc) * Td),
    "D^omegaomega": 43 * (1 - e**2) / (60 * Td) + _D_OmOm_long * sp.cos(inc) ** 2,
    "D^omegaM0": -SQ * (67 + 43 * e**2) / (60 * Td),
    "D^OmegaM0": sp.Integer(0),
    "D^iM0": sp.Integer(0),
}

# ---------------------------------------------------------------------------
# White-noise tidal regime (sec:shortcoherence)
# ---------------------------------------------------------------------------
WHITENOISE = {
    # eqn:Ba-short-coherence .. eqn:Dee-short-coherence
    "B^a": a * (18 + 19 * e**2) / (30 * Td),
    "B^e": (28 - 51 * e**2 - 103 * e**4) / (240 * e * Td),
    "D^aa": 2 * a**2 * (2 + e**2) / (15 * Td),
    "D^ae": -2 * a * e * (1 - e**2) / (15 * Td),
    "D^ee": 7 * (1 - e**2) * (4 + 5 * e**2) / (120 * Td),
    # eqn:BE-short .. eqn:DJJ-short
    "B^E": -EE * (2 + 3 * e**2) / (6 * Td),
    "B^J": JJ * (8 + 40 * e**2 + 15 * e**4) / (240 * (1 - e**2) * Td),
    "D^EE": 2 * EE**2 * (2 + e**2) / (15 * Td),
    "D^EJ": -EE * JJ * (2 + 3 * e**2) / (15 * Td),
    "D^JJ": G * m * a * (8 + 40 * e**2 + 15 * e**4) / (120 * Td),
    # body frame, eqn:Dhatehate-short .. eqn:DM0M0-short
    "D^ee_hat": (4 + 41 * e**2 + 18 * e**4) / (120 * (1 - e**2) * Td),
    "D^qq_hat": (4 + 3 * e**2) / (120 * Td),
    "D^JJ_hat": (28 + 25 * e**2 - 46 * e**4) / (120 * e**2 * Td),
    "D^JM0_hat": -SQ * (28 + 145 * e**2 + 44 * e**4) / (120 * e**2 * Td),
    "D^M0M0": (28 + 333 * e**2 + 349 * e**4 + 46 * e**6) / (120 * e**2 * Td),
    "B^M0": sp.Integer(0),
}

# Appendix B, white-noise block
_D_OmOm_short = (8 + 5 * e**2 * (8 + 3 * e**2) - 21 * e**2 * (2 + e**2) * sp.cos(2 * om)) \
    / (240 * (1 - e**2) * sp.sin(inc) ** 2 * Td)
WHITENOISE_EULER = {
    "B^Omega": -7 * e**2 * (2 + e**2) * sp.sin(2 * om)
    / (80 * (1 - e**2) * sp.tan(inc) * sp.sin(inc) * Td),
    "B^i": (8 + 5 * e**2 * (8 + 3 * e**2) - 21 * e**2 * (2 + e**2) * sp.cos(2 * om))
    / (480 * (1 - e**2) * sp.tan(inc) * Td),
    "B^omega": 7 * e**2 * (2 + e**2) * (3 + sp.cos(2 * inc)) * sp.sin(2 * om)
    / (320 * (1 - e**2) * sp.sin(inc) ** 2 * Td),
    "D^OmegaOmega": _D_OmOm_short,
    "D^Omegai": 7 * e**2 * (2 + e**2) * sp.sin(2 * om) / (80 * (1 - e**2) * sp.sin(inc) * Td),
    "D^Omegaomega": -_D_OmOm_short * sp.cos(inc),
    "D^ii": (8 + 5 * e**2 * (8 + 3 * e**2) + 21 * e**2 * (2 + e**2) * sp.cos(2 * om))
    / (240 * (1 - e**2) * Td),
    "D^iomega": -7 * e**2 * (2 + e**2) * sp.sin(2 * om) / (80 * (1 - e**2) * sp.tan(inc) * Td),
    "D^omegaomega": (28 + 25 * e**2 - 46 * e**4) / (120 * e**2 * Td)
    + _D_OmOm_short * sp.cos(inc) ** 2,
    "D^omegaM0": -SQ * (28 + 145 * e**2 + 44 * e**4) / (120 * e**2 * Td),
    "D^OmegaM0": sp.Integer(0),
    "D^iM0": sp.Integer(0),
}

# ---------------------------------------------------------------------------
# Traceless (GW) correlator, sec:blas
# ---------------------------------------------------------------------------
TRACELESS_ADIABATIC = {
    # eqn:DhatJhatJ-traceless block (adiabatic)
    "D^JJ_hat": 7 * (1 - e**2) / (15 * Td),
    "D^JM0_hat": -SQ * (8 + 7 * e**2) / (15 * Td),
    "D^M0M0": (31 + 48 * e**2 + 21 * e**4) / (45 * Td),
}
TRACELESS_WHITENOISE = {
    "B^a": 11 * a * (2 + e**2) / (45 * Td),
    "B^E": -EE * (2 + 3 * e**2) / (9 * Td),
    "B^e": (64 - 213 * e**2 - 229 * e**4) / (720 * e * Td),
    "D^aa": 4 * a**2 * (3 - e**2) / (45 * Td),
    "D^EE": 4 * EE**2 * (3 - e**2) / (45 * Td),
    "D^ae": -11 * a * e * (1 - e**2) / (45 * Td),
    "D^ee": (64 + 61 * e**2 - 125 * e**4) / (360 * Td),
    "D^JJ_hat": (64 + 15 * e**2 - 58 * e**4) / (360 * e**2 * Td),
    "D^JM0_hat": -SQ * (64 + 175 * e**2 + 62 * e**4) / (360 * e**2 * Td),
    "D^M0M0": (64 + 399 * e**2 + 487 * e**4 + 58 * e**6) / (360 * e**2 * Td),
}

# ---------------------------------------------------------------------------
# Impulsive encounters: point-mass closed forms (sec:point-mass)
# eqn:Ba-point-mass .. eqn:DM0M0-point-mass, plus the (E, J) block.
# ---------------------------------------------------------------------------
POINT_MASS = {
    "B^a": a / (15 * Tds) * (7 * L - sp.Rational(32, 3) - 6 * SQ),
    "B^e": 1 / (15 * Tds) * (5 * (1 - 3 * e**2) / (4 * e) * L
                             + (33 * e**4 - 29 * e**2 + 2) / (12 * e**3)
                             + (34 * e**4 - 3 * e**2 - 1) / (6 * e**3) * SQ),
    "D^aa": a**2 / (15 * Tds) * (4 * L - sp.Rational(32, 3)),
    "D^ae": a / (15 * Tds) * (-4 * (1 - e**2) * (1 - SQ) / e),
    "D^ee": 1 / (15 * Tds) * (sp.Rational(5, 2) * (1 - e**2) * L
                              - (1 - e**2) * (e**2 + 2) / (6 * e**2)
                              - (1 - e**2) * (16 * e**2 - 1) / (3 * e**2) * SQ),
    "D^ee_hat": 1 / (15 * Tds) * ((4 * e**2 + 1) / (2 * (1 - e**2)) * L
                                  - (14 * e**4 - 13 * e**2 + 2) / (6 * e**2 * (1 - e**2))
                                  - (6 * e**4 + 10 * e**2 - 1) / (3 * e**2 * (1 - e**2)) * SQ),
    "D^qq_hat": 1 / (15 * Tds) * (L / 2 - (5 * e**2 - 2) / (6 * e**2)
                                  - (2 * e**2 + 1) / (3 * e**2) * SQ),
    "D^JJ_hat": 1 / (15 * Tds) * ((5 - 4 * e**2) / (2 * e**2) * L
                                  + (2 * e**4 - 7 * e**2 + 2) / (6 * e**4)
                                  + (12 * e**4 - 14 * e**2 - 1) / (3 * e**4) * SQ),
    "D^JM0_hat": 1 / (15 * Tds) * (-(2 * e**2 + 5) * SQ / (2 * e**2) * L
                                   - (20 * e**4 - 19 * e**2 - 1) / (3 * e**4)
                                   - (20 * e**4 + 5 * e**2 + 2) / (6 * e**4) * SQ),
    "D^M0M0": 1 / (15 * Tds) * ((4 * e**4 + 11 * e**2 + 5) / (2 * e**2) * L
                                + (-2 * e**6 + 93 * e**4 + 15 * e**2 + 2) / (6 * e**4)
                                - (4 * e**6 + 30 * e**4 + 25 * e**2 + 1) / (3 * e**4) * SQ),
    # (E, J) block of sec:point-mass
    "B^E": -2 * EE / (15 * Tds) * (sp.Rational(3, 2) * L - 3 * SQ),
    "D^EE": 4 * EE**2 / (15 * Tds) * (L - sp.Rational(8, 3)),
    "D^EJ": -2 * EE * JJ / (15 * Tds) * (L - sp.Rational(2, 3) - 2 * SQ),
    "D^JJ": -2 * EE * a**2 / (15 * Tds) * ((1 + sp.Rational(3, 2) * e**2) * L
                                           + 1 - sp.Rational(3, 2) * e**2
                                           - (4 * e**2 + 11) / 3 * SQ),
}
# B^J = D^JJ/(2J), stated in the same block
POINT_MASS["B^J"] = POINT_MASS["D^JJ"] / (2 * JJ)

# eqn:Td*: 1/(15 T_d^*) relation to T_d (both directions are checked)
TDSTAR_BRACKET = LogLam - sp.Rational(8, 3) + 4 * sp.log(2 * SQ / (1 + SQ))

# eqn:f-ss-point-mass correction chi(e): f_ss ~ 2e/a^2 (1 + chi/logLambda)
F_SS_PM_CHI = 4 * SQ - 2 * sp.log(1 + SQ)
