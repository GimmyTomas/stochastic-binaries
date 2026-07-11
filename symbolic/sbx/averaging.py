"""Fast orbit averaging.

The paper defines the orbit average of a function X(phi, E) as (eqn:orbit-average)

    Xbar = (1/2pi) Int_0^{2pi} X(phi(E), E) (1 - e cos E) dE ,

i.e. an average over the eccentric anomaly E with Jacobian U = 1 - e*cos(E).

Naive ``sympy.integrate`` is far too slow for the expressions that appear in
the Fokker--Planck coefficient calculations. Instead, every expression built
by this package is a *finite sum of monomials*

    coeff * cos(E)^c * sin(E)^s * U^q * LNU^l        (l in {0, 1})

with coeff independent of E. The average of each monomial is a table lookup:

* odd powers of sin(E) average to zero;
* even powers of sin(E) are rewritten via sin^2 = 1 - cos^2;
* cos(E)^c is rewritten via cos(E) = (1 - U)/e, leaving pure powers of U;
* <U^q>            : binomial expansion + Wallis integrals   (q >= 0)
                     closed-form recursion                    (q < 0)
* <U^q log U>      : closed forms derived from the classical Fourier series
                     log(1-e cos E) = -log(1+beta^2) - 2 Sum_k (beta^k/k) cos kE,
                     1/(1-e cos E)  = (1/w) (1 + 2 Sum_k beta^k cos kE),
                     with beta = (1 - w)/e = e/(1 + w),  w = sqrt(1-e^2).

Every table entry is verified numerically by ``tests_averaging.py`` (50-digit
mpmath quadrature at random eccentricities).
"""

import sympy as sp

from .symbols import a, e, E, U, LNU, QRS, QTS

_cos = sp.cos(E)
_sin = sp.sin(E)
_SQ = sp.sqrt(1 - e**2)


# ---------------------------------------------------------------------------
# Elementary moment tables
# ---------------------------------------------------------------------------

def wallis(mpow):
    """<cos(E)^m> = binom(m, m/2)/2^m for even m, 0 for odd m."""
    mpow = int(mpow)
    if mpow % 2:
        return sp.Integer(0)
    return sp.binomial(mpow, mpow // 2) / sp.Integer(2) ** mpow


_U_POS_CACHE = {}


def avg_U(q):
    """<U^q> for integer q >= 0 (binomial + Wallis)."""
    q = int(q)
    if q not in _U_POS_CACHE:
        _U_POS_CACHE[q] = sp.expand(
            sum(sp.binomial(q, k) * (-e) ** k * wallis(k) for k in range(q + 1))
        )
    return _U_POS_CACHE[q]


_U_NEG_CACHE = {1: 1 / _SQ}


def avg_U_inv(p):
    """<U^{-p}> for integer p >= 1.

    Recursion: with A_j = <U^{-j}>,
        dA_j/de = j <cos E * U^{-(j+1)}> = (j/e) (A_{j+1} - A_j)
    hence A_{j+1} = A_j + (e/j) * dA_j/de,  starting from A_1 = 1/sqrt(1-e^2).
    """
    p = int(p)
    jmax = max(_U_NEG_CACHE)
    while jmax < p:
        Aj = _U_NEG_CACHE[jmax]
        _U_NEG_CACHE[jmax + 1] = sp.simplify(Aj + e * sp.diff(Aj, e) / jmax)
        jmax += 1
    return _U_NEG_CACHE[p]


# --- log moments -------------------------------------------------------------
#
# beta = (1 - w)/e with w = sqrt(1-e^2). Then
#   U = (1 + beta^2 - 2 beta cos E) / (1 + beta^2),
#   log U = -log(1+beta^2) - 2 Sum_{k>=1} (beta^k / k) cos kE,
#   1/U   = (1/w) (1 + 2 Sum_{k>=1} beta^k cos kE).
#
# <log U>            = -log(1+beta^2) + 0                = log((1+w)/2)   [classical]
# <U^{-1} log U>     = (1/w) [ -log(1+beta^2) - 2 Sum_k beta^{2k}/k ]
#                    = (1/w) [ -log(1+beta^2) + 2 log(1-beta^2) ]
# For q >= 1, U^q is a finite cosine polynomial; writing
#   U^q = <U^q> + Sum_{l=1}^{q} h_l(q) * 2 cos(lE)   (h_l = <U^q cos lE>),
#   <U^q log U> = -log(1+beta^2) <U^q> - 2 Sum_{l=1}^{q} (beta^l / l) h_l(q).
# Deeper negative moments follow from the derivative recursion
#   Lam_{q-1} = Lam_q - (e/q) dLam_q/de - (1/q) (<U^{q-1}> - <U^q>),  q != 0,
# applied at q = -1 to obtain Lam_{-2} from Lam_{-1}.

_beta = (1 - _SQ) / e


def _fourier_cos_coeff(expr, l):
    """<expr * cos(lE)> for expr a real trig polynomial in E.

    Writing expr = Sum_k c_k exp(i k E), one has <expr cos(lE)> = (c_l + c_{-l})/2.
    """
    z = sp.Dummy("z")
    x = sp.expand_trig(expr).rewrite(sp.exp).expand()
    x = sp.expand(x.subs({sp.exp(sp.I * E): z, sp.exp(-sp.I * E): 1 / z}))
    c_pos = c_neg = sp.S.Zero
    for term in sp.Add.make_args(x):
        if not term.has(z):
            k = 0
        else:
            # term = coeff * z**k (k may be negative)
            k = sp.degree(sp.together(term * z**64), z) - 64
        if k == l:
            c_pos += term / z**k
        elif k == -l:
            c_neg += term / z**k
    if l == 0:
        return sp.simplify(c_pos)
    return sp.simplify((c_pos + c_neg) / 2)


_LOG_CACHE = {}


def avg_U_log(q):
    """<U^q * log U> for any integer q.

    q = 0, -1 are closed forms from the Fourier-series products; deeper
    negative moments follow from the derivative recursion (see module
    docstring)  Lam_{p-1} = Lam_p - (e/p) dLam_p/de - (1/p)(U_{p-1} - U_p),
    valid for p != 0; positive moments from the finite cosine expansion.
    """
    q = int(q)
    if q in _LOG_CACHE:
        return _LOG_CACHE[q]
    beta = _beta
    # With beta = (1-w)/e, w = sqrt(1-e^2), the two series constants reduce to
    #   1 + beta^2 = 2(1-w)/e^2 = 2/(1+w),   1 - beta^2 = 2w/(1+w),
    # so every log moment lives in the atom basis {log 2, log w, log(1+w)}.
    log_1pb2 = sp.log(2 / (1 + _SQ))
    log_1mb2 = sp.log(2 * _SQ / (1 + _SQ))

    def _avg_pow(x):
        return avg_U(x) if x >= 0 else avg_U_inv(-x)

    if q == 0:
        val = sp.log((1 + _SQ) / 2)
    elif q == -1:
        val = (-log_1pb2 + 2 * log_1mb2) / _SQ
    elif q <= -2:
        p = q + 1  # p in [-1, ...], p != 0
        lam_p = avg_U_log(p)
        val = lam_p - e * sp.diff(lam_p, e) / p - (_avg_pow(q) - _avg_pow(p)) / p
    else:  # q >= 1
        upoly = sp.expand((1 - e * _cos) ** q)
        val = -log_1pb2 * avg_U(q)
        for l in range(1, q + 1):
            h_l = _fourier_cos_coeff(upoly, l)
            val += -2 * (beta**l / l) * h_l
    val = sp.simplify(sp.radsimp(val))
    _LOG_CACHE[q] = val
    return val


# ---------------------------------------------------------------------------
# Monomial classification and the main entry point
# ---------------------------------------------------------------------------

_MONO_CACHE = {}
_FACTOR_CACHE = {}


def moment_symbol(flavor, q):
    """Opaque symbol for (1/2pi) Int U^q * Q_flavor(U) dE (flavor in 'r','t').

    Used to express general-perturber (impulsive) results without specifying
    the kick covariances Q_r(r), Q_t(r): two expressions are equal as linear
    functionals of (Q_r, Q_t) iff their canonical moment-symbol forms agree.
    """
    tag = f"m{-q}" if q < 0 else str(q)
    return sp.Symbol(f"MQ{flavor}_{tag}")


def _avg_monomial(cpow, spow, qpow, logpow, flavor=None):
    """<cos^c sin^s U^q LNU^l [Q_flavor(U)]> with l in {0,1}."""
    key = (cpow, spow, qpow, logpow, flavor)
    if key in _MONO_CACHE:
        return _MONO_CACHE[key]
    if spow % 2:
        _MONO_CACHE[key] = sp.Integer(0)
        return sp.Integer(0)
    if flavor is not None and logpow:
        raise NotImplementedError("log moments of opaque Q not needed/supported")
    # sin^s -> (1 - cos^2)^{s/2}; then cos^m -> ((1-U)/e)^m; collect U powers
    total = sp.S.Zero
    sin_expand = sp.Poly(sp.expand((1 - sp.Symbol("_c") ** 2) ** (spow // 2)),
                         sp.Symbol("_c"))
    for (deg_c,), coeff_c in sin_expand.terms():
        mm = cpow + deg_c
        # cos^mm = ((1-U)/e)^mm
        v = sp.Symbol("_v")
        cos_expand = sp.Poly(sp.expand((1 - v) ** mm), v)
        for (deg_v,), coeff_v in cos_expand.terms():
            qq = qpow + deg_v
            if flavor is not None:
                mom = moment_symbol(flavor, qq)
            elif logpow == 0:
                mom = avg_U(qq) if qq >= 0 else avg_U_inv(-qq)
            else:
                mom = avg_U_log(qq)
            total += coeff_c * coeff_v * mom / e**mm
    total = sp.simplify(sp.radsimp(sp.together(total)))
    _MONO_CACHE[key] = total
    return total


def orbit_average(expr, jacobian=True):
    """Orbit average (eqn:orbit-average) of a U-form expression.

    ``expr`` must be a finite sum of terms coeff * cos(E)^c * sin(E)^s * U^q
    * LNU^l with coeff independent of E, U, LNU (l in {0, 1}).
    If ``jacobian`` is True the measure (1 - e cos E) dE/(2 pi) is used,
    i.e. the expression is multiplied by U before averaging.
    """
    if jacobian:
        expr = expr * U
    if any(isinstance(f, sp.Function) and f.args and f.args[0].has(E) and f.args[0] != E
           for f in expr.atoms(sp.Function)):
        expr = sp.expand_trig(expr)  # multi-angle trig of E (rare)
    expr = sp.expand(expr)
    pieces = []
    for term in sp.Add.make_args(expr):
        cpow = spow = qpow = logpow = 0
        flavor = None
        rest = []
        stack = list(sp.Mul.make_args(term))
        while stack:
            fac = stack.pop()
            base, p = fac.as_base_exp()
            if base == _cos:
                cpow += p
            elif base == _sin:
                spow += p
            elif base == U:
                qpow += p
            elif base == LNU:
                logpow += p
            elif base == QRS or base == QTS:
                if flavor is not None or p != 1:
                    raise ValueError(f"only terms linear in Q_r/Q_t supported: {term}")
                flavor = "r" if base == QRS else "t"
            elif fac.has(E) or fac.has(U) or fac.has(LNU):
                # e.g. a Pow whose base is a merged product like G*m*U**2*(1-e**2):
                # factor the base (cached) and process each factor separately
                # ((x*y)^p = x^p y^p is safe: bases are positive/real here).
                if base not in _FACTOR_CACHE:
                    _FACTOR_CACHE[base] = sp.factor(base)
                factored = _FACTOR_CACHE[base]
                if isinstance(factored, (sp.Mul, sp.Pow)) and factored != base:
                    stack.extend(f ** p for f in sp.Mul.make_args(factored))
                else:
                    raise ValueError(f"cannot classify factor {fac} in term {term}")
            else:
                rest.append(fac)
        if not (cpow == int(cpow) and spow == int(spow) and qpow == int(qpow)):
            raise ValueError(f"non-integer powers in term {term}")
        if logpow not in (0, 1):
            raise ValueError(f"log power {logpow} not in (0,1) in term {term}")
        mom = _avg_monomial(int(cpow), int(spow), int(qpow), int(logpow), flavor)
        if mom != 0:
            pieces.append(sp.Mul(*rest) * mom)
    return sp.Add(*pieces)


def to_explicit(expr):
    """Substitute U -> 1 - e*cos(E) and LNU -> log(1 - e*cos(E)) (for numerics)."""
    return expr.subs({LNU: sp.log(1 - e * _cos), U: 1 - e * _cos})
