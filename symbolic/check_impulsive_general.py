"""Impulsive encounters with a GENERIC spherical perturber (sec:impulsive-setup
-- sec:coefficient-impulsive): verify the drift and diffusion coefficients as
functionals of the kick covariances Q_r(r), Q_t(r), and derive the Euler-angle
sector that the paper does not write out.

Method. The phase-space force is F^mu = G^mu_alpha Dv^alpha (eqn:F-impulsive)
with G^mu_alpha read off Gauss's equations, and (eqn:Bmu-impulsive,
eqn:Dmunu-impulsive)

    B^mu   = (1/2) <d_nu(G^mu_j) G^nu_k Q^{jk}>_orb ,
    D^munu = <G^mu_j G^nu_k Q^{jk}>_orb .

Here G^mu_j = G^mu_alpha ehat^alpha_j is the SPACE-FRAME force vector: the
kick Delta-v is fixed in space, so the slow-variable derivative at fixed fast
phase (eqn:dEdedM0) acts on the Gauss coefficients AND on the orbital-frame
unit vectors ehat^alpha(w) (it does not hit the kick covariances themselves).
The frame-rotation terms are essential for the drifts: dropping them loses
exactly the <Q_t>-type contributions in B^a, B^e, B^E, B^J. In the orbital
frame the isotropic-bath covariance is Q^{alpha beta} = diag(Q_r, Q_t, Q_t),
i.e. in space Q^{jk} = Q_t delta_jk + (Q_r - Q_t) rhat_j rhat_k. Both the
computed coefficients and the draft's quoted orbit-average expressions
(eqn:Ba-impulsive .. eqn:Dee-impulsive, the (E,J) block, and
eqn:Dhatehate-impulsive .. eqn:DM0M0-impulsive) are reduced to a canonical
form in the opaque moment symbols

    MQx_q = (1/2pi) Int (1 - e cos E)^q Q_x dE ,     x in {r, t} ,

so that equality as linear functionals of (Q_r, Q_t) is checked exactly.

The Euler-angle sector (Omega, i, omega, M0) for generic Q -- NOT given in the
paper -- is derived here, checked for internal consistency (round trip to the
body frame, D^{Omega omega} = -D^{Omega Omega} cos i, D^{Omega M0} = D^{i M0}
= 0), and written to output/impulsive-euler-coefficients.tex.
"""

import pathlib
import sys
import time

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sbx.symbols import (a, e, G, m, E, U, Om, inc, om, M0, SLOW_VARS,
                         QRS, QTS, SQ, ANGLE_TO_SYM, c_i)
from sbx.averaging import orbit_average
from sbx.anomalies import d_fixed_phase, cosphi, sinphi, r_of_E
from sbx.gauss import gauss_coefficients, FRAME_PF
from sbx.frames import body_frame
from sbx.pipelines import _generators
from sbx.verify import assert_eq, summary, canon
from sbx import targets

t0 = time.time()
OUT_DIR = pathlib.Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# Dimensionless velocity components (draft notation: v~_r, v~_phi; r~ = U)
vr_t = e * sp.sin(E) / U
vphi_t = SQ / U
JJ_expr = sp.sqrt(G * m * a * (1 - e**2))
EE_expr = -G * m / (2 * a)


def oa(expr):
    """Orbit average (with Jacobian), angles as symbols, moment-symbol form."""
    return orbit_average(sp.expand(expr).xreplace(ANGLE_TO_SYM))


# ---------------------------------------------------------------------------
# Space-frame vector representation. Every force vector is a sum
# G^mu = sum_alpha c^mu_alpha * (R v_alpha) with v_alpha the perifocal frame
# vectors; rotations drop out of dot products, and orientation derivatives
# enter through the generators K_nu = R^T dR/dnu (as in sbx.pipelines).
# ---------------------------------------------------------------------------
KGEN = _generators()


def vec_G(rows):
    return [(c, v) for c, v in zip(rows, FRAME_PF) if c != 0]


def vec_dG(rows, nu):
    """Fixed-phase derivative of the space-frame force vector, in perifocal
    representation: coefficients, frame vectors (through phi(E, e)), and the
    rotation R (through K_nu) are all differentiated."""
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


def q_bilinear_vec(X, Y):
    """x_j y_k Q^{jk} with Q^{jk} = Q_t d_jk + (Q_r - Q_t) rhat_j rhat_k."""
    v_r = FRAME_PF[0]
    pieces = []
    for c1, u1 in X:
        for c2, u2 in Y:
            pieces.append(sp.expand(c1 * c2 * (
                QTS * u1.dot(u2) + (QRS - QTS) * u1.dot(v_r) * u2.dot(v_r))))
    return sp.Add(*pieces)


# ---------------------------------------------------------------------------
# Compute B^mu, D^{mu nu} for the six slow variables from the formalism
# ---------------------------------------------------------------------------
print("Computing impulsive B, D for generic Q (moment-symbol form)...")
G_rows = {var: gauss_coefficients(var) for var in SLOW_VARS}
D = {}
for i_mu, mu in enumerate(SLOW_VARS):
    for nu in SLOW_VARS[i_mu:]:
        D[(mu, nu)] = oa(q_bilinear_vec(vec_G(G_rows[mu]), vec_G(G_rows[nu])))
        D[(nu, mu)] = D[(mu, nu)]
B = {}
for mu in SLOW_VARS:
    val = sp.S.Zero
    for nu in SLOW_VARS:
        val += oa(q_bilinear_vec(vec_dG(G_rows[mu], nu), vec_G(G_rows[nu])))
    B[mu] = sp.expand(val / 2)
print(f"  done ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# (a, e) sector: eqn:Ba-impulsive .. eqn:Dee-impulsive
# ---------------------------------------------------------------------------
print("(a,e) sector vs the draft's orbit-average expressions:")
tgt_Ba = a**2 / (G * m) * oa(QRS + 2 * QTS + 4 * (vr_t**2 * QRS + vphi_t**2 * QTS))
tgt_Be = a / (G * m) * oa((1 - e**2) / (2 * e) * (QRS + 2 * QTS)
                          - (1 - e**2) ** 2 / (2 * e**3) * (vr_t**2 * QRS + vphi_t**2 * QTS)
                          - (1 + e**2) / (2 * e**3) * U**2 * QTS
                          + (1 - e**4) / e**3 * QTS)
tgt_Daa = 4 * a**3 / (G * m) * oa(vr_t**2 * QRS + vphi_t**2 * QTS)
tgt_Dae = 2 * a**2 * (1 - e**2) / (G * m * e) * oa(vr_t**2 * QRS + vphi_t**2 * QTS - QTS)
tgt_Dee = a / (G * m) * oa((1 - e**2) ** 2 / e**2 * (vr_t**2 * QRS + vphi_t**2 * QTS)
                           + (1 - e**2) / e**2 * U**2 * QTS
                           - 2 * (1 - e**2) ** 2 / e**2 * QTS)
assert_eq("B^a (eqn:Ba-impulsive)", B[a], tgt_Ba)
assert_eq("B^e", B[e], tgt_Be)
assert_eq("D^aa", D[(a, a)], tgt_Daa)
assert_eq("D^ae", D[(a, e)], tgt_Dae)
assert_eq("D^ee (eqn:Dee-impulsive)", D[(e, e)], tgt_Dee)

print("Block structure:")
offblock = [D[(a, v)] for v in (Om, inc, om, M0)] + [D[(e, v)] for v in (Om, inc, om, M0)]
assert_eq("D^{a,rest} = D^{e,rest} = 0", sum(sp.Abs(sp.simplify(x)) for x in offblock), 0)
assert_eq("D^{Omega M0} = 0", D[(Om, M0)], 0)
assert_eq("D^{i M0} = 0", D[(inc, M0)], 0)

# ---------------------------------------------------------------------------
# (E, J) sector, computed directly from G^E = (v_r, v_phi, 0), G^J = (0, r, 0)
# ---------------------------------------------------------------------------
print("(E,J) sector vs the draft:")
vr_full = sp.sqrt(G * m / a) * vr_t
vphi_full = sp.sqrt(G * m / a) * vphi_t
G_E = (vr_full, vphi_full, sp.Integer(0))
G_J = (sp.Integer(0), r_of_E, sp.Integer(0))
rows_all = dict(G_rows)
rows_all["E"] = G_E
rows_all["J"] = G_J

BE = sp.S.Zero
BJ = sp.S.Zero
for nu in SLOW_VARS:
    BE += oa(q_bilinear_vec(vec_dG(G_E, nu), vec_G(G_rows[nu])))
    BJ += oa(q_bilinear_vec(vec_dG(G_J, nu), vec_G(G_rows[nu])))
BE, BJ = sp.expand(BE / 2), sp.expand(BJ / 2)
DEE = oa(q_bilinear_vec(vec_G(G_E), vec_G(G_E)))
DEJ = oa(q_bilinear_vec(vec_G(G_E), vec_G(G_J)))
DJJ = oa(q_bilinear_vec(vec_G(G_J), vec_G(G_J)))
assert_eq("B^E = <Q_r + 2Q_t>/2", BE, oa(QRS + 2 * QTS) / 2)
assert_eq("B^J = <r^2 Q_t>/(2J)", BJ, oa(r_of_E**2 * QTS) / (2 * JJ_expr))
assert_eq("D^EE = -2E <v~_r^2 Q_r + v~_phi^2 Q_t>", DEE,
          -2 * EE_expr * oa(vr_t**2 * QRS + vphi_t**2 * QTS))
assert_eq("D^EJ = J <Q_t>", DEJ, JJ_expr * oa(QTS))
assert_eq("D^JJ = <r^2 Q_t>", DJJ, oa(r_of_E**2 * QTS))
assert_eq("B^J = D^JJ/(2J) (App C identity, generic Q)", BJ, sp.expand(DJJ / (2 * JJ_expr)))

# ---------------------------------------------------------------------------
# Orientation sector: body frame vs eqn:Dhatehate-impulsive .. eqn:DM0M0-impulsive
# ---------------------------------------------------------------------------
print("Body-frame orientation sector vs the draft:")
B_eul = [B[Om], B[inc], B[om]]
D_eul = sp.Matrix(3, 3, lambda p, q: D[((Om, inc, om)[p], (Om, inc, om)[q])])
D_eulM0 = [D[(Om, M0)], D[(inc, M0)], D[(om, M0)]]
B_hat, D_hat, D_hatM0 = body_frame(B_eul, D_eul, D_eulM0)

# Draft targets (phi-averages converted to E-form):
# cos(phi), sin(phi) from sbx.anomalies; (2 + e cos phi)/(1 + e cos phi)
# = (2 - e^2 - e cos E)/(1 - e^2); 1/(1 + e cos phi) = U/(1 - e^2).
ratio = (2 - e**2 - e * sp.cos(E)) / (1 - e**2)
tgt_Dee_hat = oa(r_of_E**2 * cosphi**2 * QTS) / (G * m * a * (1 - e**2))
tgt_Dqq_hat = oa(r_of_E**2 * sinphi**2 * QTS) / (G * m * a * (1 - e**2))
tgt_DJJ_hat = a * (1 - e**2) / (G * m * e**2) * oa(cosphi**2 * QRS + ratio**2 * sinphi**2 * QTS)
tgt_DJM0 = -a * (1 - e**2) ** sp.Rational(3, 2) / (G * m * e) * oa(
    (cosphi / e - 2 * U / (1 - e**2)) * cosphi * QRS + ratio**2 * sinphi**2 * QTS / e)
tgt_DM0M0 = a * (1 - e**2) ** 2 / (G * m) * oa(
    (cosphi / e - 2 * U / (1 - e**2)) ** 2 * QRS + ratio**2 * sinphi**2 * QTS / e**2)

assert_eq("B^ehat = 0", B_hat[0], 0)
assert_eq("B^qhat = 0", B_hat[1], 0)
assert_eq("B^Jhat = 0", B_hat[2], 0)
assert_eq("B^M0 = 0", B[M0], 0)
assert_eq("D^ee_hat (eqn:Dhatehate-impulsive)", D_hat[0, 0], tgt_Dee_hat)
assert_eq("D^qq_hat", D_hat[1, 1], tgt_Dqq_hat)
assert_eq("D^JJ_hat", D_hat[2, 2], tgt_DJJ_hat)
assert_eq("D^JM0_hat (eqn:DM0hatJ-impulsive)", D_hatM0[2], tgt_DJM0)
assert_eq("D^M0M0 (eqn:DM0M0-impulsive)", D[(M0, M0)], tgt_DM0M0)
assert_eq("D^eq_hat = 0", D_hat[0, 1], 0)
assert_eq("D^eJ_hat = 0", D_hat[0, 2], 0)
assert_eq("D^qJ_hat = 0", D_hat[1, 2], 0)
assert_eq("D^eM0_hat = 0", D_hatM0[0], 0)
assert_eq("D^qM0_hat = 0", D_hatM0[1], 0)

# ---------------------------------------------------------------------------
# NEW RESULT: Euler-angle sector for generic Q (not in the paper)
# ---------------------------------------------------------------------------
print("Euler-angle sector for generic Q (new; structural checks + LaTeX output):")
assert_eq("D^Omegaomega = -D^OmegaOmega cos(i) (generic Q)",
          D[(Om, om)], -D[(Om, Om)] * c_i)

MOM_LATEX = {}
for flav in ("r", "t"):
    for q in range(-3, 7):
        tag = f"m{-q}" if q < 0 else str(q)
        sym = sp.Symbol(f"MQ{flav}_{tag}")
        # MQx_q = (1/2pi) Int U^q Q_x dE = overline{ U^(q-1) Q_x }  (the
        # draft's overline includes the Jacobian U)
        power = q - 1
        if power == 0:
            MOM_LATEX[sym] = sp.Symbol(rf"\overline{{Q_{flav}}}")
        else:
            MOM_LATEX[sym] = sp.Symbol(rf"\overline{{\tilde r^{{{power}}} Q_{flav}}}")


def to_latex(expr):
    """Render as (1/denominator) * sum_over_moments coeff * moment."""
    from sbx.symbols import s_i, c_w, s_w, w as w_sym
    expr = sp.cancel(sp.together(sp.simplify(canon(expr))))
    num, den = sp.fraction(expr)
    moms = sorted([s for s in num.free_symbols if str(s).startswith("MQ")],
                  key=str)
    num = sp.collect(sp.expand(num), moms)
    pieces = []
    for mom in moms:
        coeff = num.coeff(mom)
        if coeff == 0:
            continue
        pieces.append((mom, sp.factor(coeff)))
    back = {c_i: sp.cos(inc), s_i: sp.sin(inc), c_w: sp.cos(om),
            s_w: sp.sin(om), w_sym: sp.sqrt(1 - e**2)}

    def tex(x):
        return sp.latex(x.xreplace(MOM_LATEX).xreplace(back))

    num_tex = " + ".join(rf"\left[{tex(cf)}\right] {tex(mm)}" for mm, cf in pieces)
    den_f = sp.factor(den)
    return rf"\frac{{ {num_tex} }}{{ {tex(den_f)} }}"


euler_results = [
    (r"B^\Omega", B[Om]), (r"B^i", B[inc]), (r"B^\omega", B[om]),
    (r"D^{\Omega\Omega}", D[(Om, Om)]), (r"D^{\Omega i}", D[(Om, inc)]),
    (r"D^{\Omega\omega}", D[(Om, om)]), (r"D^{ii}", D[(inc, inc)]),
    (r"D^{i\omega}", D[(inc, om)]), (r"D^{\omega\omega}", D[(om, om)]),
    (r"D^{\omega M_0}", D[(om, M0)]), (r"D^{M_0 M_0}", D[(M0, M0)]),
]
tex_path = OUT_DIR / "impulsive-euler-coefficients.tex"
with open(tex_path, "w") as fh:
    fh.write("% Euler-angle-sector Fokker--Planck coefficients for impulsive\n"
             "% encounters with a GENERIC spherical perturber (kick covariances\n"
             "% Q_r(r), Q_t(r)); generated by check_impulsive_general.py.\n"
             "% Notation: overline{X} = (1/2pi) Int X (1 - e cos E) dE is the\n"
             "% orbit average (eqn:orbit_average); tilde-r = 1 - e cos E = r/a.\n"
             "% These expressions are NOT in the paper; they were derived with\n"
             "% the same machinery that reproduces every published coefficient,\n"
             "% and transform to the paper's body-frame results via eqn:Matrix.\n"
             "\\begin{align}\n")
    lines = [f"{name} &= {to_latex(expr)}" for name, expr in euler_results]
    fh.write(" ,\\\\\n".join(lines))
    fh.write(" .\n\\end{align}\n")
print(f"  wrote {tex_path}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
sys.exit(summary("check_impulsive_general"))
