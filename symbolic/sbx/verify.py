"""Equality assertions with canonicalization and a high-precision numeric fallback.

``assert_eq(name, computed, target)`` canonicalizes the difference by

1. expanding multiple angles (cos(2*omega) -> ..., tan -> sin/cos);
2. replacing (1-e^2)^(p/2) by w^p, with w an independent positive symbol;
3. replacing sin/cos of the Euler angles by plain symbols;
4. expanding logarithms into the independent atoms
   {log(w), log(1+w), log(2), log(e), logLambda, L_a, ...} and replacing each
   atom by a fresh symbol;
5. reducing even powers via w^2 -> 1-e^2, sin^2 -> 1-cos^2;
6. checking that the resulting rational function is identically zero.

If the symbolic reduction does not reach zero, both sides are evaluated with
50-digit mpmath at many random rational points; agreement at that precision is
reported as "NUM" (numerically verified, symbolically unresolved) rather than
a PASS, so a physics error can never hide behind a SymPy weakness.
"""

import random

import mpmath as mp
import sympy as sp

from .symbols import (a, e, G, m, w, Om, inc, om, Td, Tds, LogLam, La, kappa,
                      Aamp, sigma, mstar, nden, Rs, ANGLE_TO_SYM,
                      c_i, s_i, c_w, s_w, c_O, s_O)

mp.mp.dps = 50

RESULTS = []

_ANGLE_SUBS = ANGLE_TO_SYM
_EVEN_PAIRS = [
    (w, 1 - e**2),
    (s_i, 1 - c_i**2),
    (s_w, 1 - c_w**2),
    (s_O, 1 - c_O**2),
]


def _sqrt_to_w(expr):
    """(1-e^2)^(p/2) -> w^p (p odd); also catches sqrt(1-e^2) inside products."""
    def repl(x):
        return w ** int(2 * x.exp)
    return expr.replace(
        lambda x: (x.is_Pow and x.base == 1 - e**2 and x.exp.is_Rational
                   and x.exp.q == 2),
        repl)


def _reduce_even(expr):
    """Reduce even powers of w and of the sin-angle symbols."""
    for sym, sub in _EVEN_PAIRS:
        expr = expr.replace(
            lambda x, s=sym: x.is_Pow and x.base == s and x.exp.is_Integer and abs(x.exp) >= 2,
            lambda x, s=sym, v=sub: v ** (x.exp // 2) * s ** (x.exp % 2)
            if x.exp > 0 else (1 / v) ** ((-x.exp) // 2) * s ** (-((-x.exp) % 2)))
    return expr


_LOG_ATOMS = {}


def _log_atomize(expr):
    """Canonicalize logarithms and replace each atom by a stable fresh symbol.

    Log arguments are put over a common denominator and factored so that
    expand_log(force=True) can split them (log((1+w)/2) would otherwise be
    the unsplittable log(w/2 + 1/2)); the redundant atom log(1 - w) is
    eliminated through (1-w)(1+w) = e^2.
    """
    if not expr.has(sp.log):
        return expr
    expr = expr.replace(lambda x: isinstance(x, sp.log),
                        lambda x: sp.log(sp.factor(sp.together(x.args[0]))))
    expr = sp.expand_log(expr, force=True)
    expr = expr.subs(sp.log(1 - w), 2 * sp.log(e) - sp.log(1 + w))
    expr = expr.subs(sp.log(w - 1), sp.log(-1) + 2 * sp.log(e) - sp.log(1 + w))
    logs = sorted(expr.atoms(sp.log), key=sp.default_sort_key)
    for lg in logs:
        if lg not in _LOG_ATOMS:
            _LOG_ATOMS[lg] = sp.Symbol(f"LOGATOM_{len(_LOG_ATOMS)}")
        expr = expr.xreplace({lg: _LOG_ATOMS[lg]})
    return expr


def canon(expr):
    expr = sp.expand_trig(sp.together(expr).rewrite(sp.sin))
    expr = _sqrt_to_w(sp.expand(expr))
    expr = expr.xreplace(_ANGLE_SUBS)
    expr = _log_atomize(expr)
    expr = _reduce_even(sp.expand(expr))
    expr = _reduce_even(sp.expand(expr))
    return expr


def _is_zero(expr):
    expr = sp.cancel(sp.together(expr))
    num, _ = sp.fraction(expr)
    return sp.expand(num) == 0


_NUM_SYMBOLS = [a, G, m, Td, Tds, LogLam, La, kappa, Aamp, sigma, mstar, nden, Rs]


def _numeric_check(diff, npoints=12, tol=1e-35):
    rng = random.Random(20260702)
    # angle symbols (c_i, s_i, ...) must be numerically consistent with the
    # corresponding trig functions of (Omega, i, omega)
    expr = diff.subs(w, sp.sqrt(1 - e**2))
    for _ in range(npoints):
        th = {Om: mp.mpf(rng.randint(10, 170)) / 100,
              inc: mp.mpf(rng.randint(10, 170)) / 100,
              om: mp.mpf(rng.randint(10, 170)) / 100}
        subs = {c_O: mp.cos(th[Om]), s_O: mp.sin(th[Om]),
                c_i: mp.cos(th[inc]), s_i: mp.sin(th[inc]),
                c_w: mp.cos(th[om]), s_w: mp.sin(th[om])}
        subs.update(th)
        for s in expr.free_symbols:
            if s in subs:
                continue
            if s == e:
                subs[s] = mp.mpf(rng.randint(5, 95)) / 100
            else:
                subs[s] = mp.mpf(rng.randint(2, 50)) / 10
        try:
            free = sorted(expr.free_symbols, key=str)
            val = sp.lambdify(free, expr, "mpmath")(*[subs[s] for s in free])
        except Exception:
            return False
        if abs(val) > tol:
            return False
    return True


def assert_eq(name, computed, target):
    """Record whether ``computed`` equals ``target``. Returns True on success."""
    diff = canon(computed - target)
    if _is_zero(diff):
        RESULTS.append((name, "SYM"))
        print(f"  [SYM ] {name}")
        return True
    diff2 = sp.simplify(diff)
    if diff2 == 0 or _is_zero(diff2):
        RESULTS.append((name, "SYM"))
        print(f"  [SYM ] {name}")
        return True
    if _numeric_check(computed - target):
        RESULTS.append((name, "NUM"))
        print(f"  [NUM ] {name}   (numerically verified at 50 digits; symbolic reduction incomplete)")
        return True
    RESULTS.append((name, "FAIL"))
    print(f"  [FAIL] {name}")
    print(f"         residual: {sp.simplify(diff2)}")
    return False


def summary(script_name):
    n_sym = sum(1 for _, s in RESULTS if s == "SYM")
    n_num = sum(1 for _, s in RESULTS if s == "NUM")
    n_fail = sum(1 for _, s in RESULTS if s == "FAIL")
    print(f"\n{script_name}: {n_sym} symbolic, {n_num} numeric-only, {n_fail} FAILED "
          f"(of {len(RESULTS)} checks)")
    return 1 if n_fail else 0
