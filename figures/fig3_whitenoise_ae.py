"""Figure 3 (fig:f-short-coherence): numerical solution of the (a, e) sector
of the Fokker--Planck equation (eqn:fp-ae) in the WHITE-NOISE tidal regime
(sec:shortcoherence), with the coefficients eqn:Ba-short-coherence ..
eqn:Dee-short-coherence and the T_d = (a0/a)^3 T_d0 scaling.

Setup as in the caption: square grid in (x = ln(a/a0), eps = e^2) on
[-3, 6] x [0, 1]; initial condition a narrow Gaussian (a/a0: 1 +/- 0.05,
e: 0.5 +/- 0.03); absorbing boundary at large a (binaries are unbound),
reflecting elsewhere (those boundaries are inaccessible). Panels: heatmaps of
f(a, e) at t/T_d0 = 0.1 and 3; marginal f(ln a, t) (NOT renormalized -- the
lost area is the disrupted fraction); eccentricity marginal f(e, t)
renormalized to unit area, with the constant-flux steady state
f_ss ~ e (4+5e^2)^{-36/35} (eqn:f-steady-state) overlaid.

Runtime: ~2 min (default 300x200 grid).  --fast: 150x100, ~15 s.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.style import setup, COLORS
from common.fp2d import FP2D
from common.coefficients import whitenoise_xe_coeffs

OUT = pathlib.Path(__file__).resolve().parent / "output"
SNAPS = [0.1, 0.3, 1.0, 3.0, 10.0]
HEAT_TIMES = [0.1, 3.0]


def initial_condition(solver, a0=1.0, e0=0.5, sig_a=0.05, sig_e=0.03):
    X, V = np.meshgrid(solver.u, solver.v, indexing="ij")
    a = np.exp(X)
    e = np.sqrt(np.maximum(V, 1e-12))
    f_ae = np.exp(-0.5 * (a - a0) ** 2 / sig_a**2 - 0.5 * (e - e0) ** 2 / sig_e**2)
    f_xe = f_ae * a / (2 * e)
    f_xe /= f_xe.sum() * solver.du * solver.dv
    return f_xe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)

    Nx, Ne, dt = (150, 100, 3e-2) if args.fast else (300, 200, 1.5e-2)
    x_edges = np.linspace(-3, 6, Nx + 1)
    v_edges = np.linspace(0, 1, Ne + 1)
    solver = FP2D(x_edges, v_edges, whitenoise_xe_coeffs(),
                  bc=("reflect", "absorb", "reflect", "reflect"))
    f0 = initial_condition(solver)
    print(f"grid {Nx}x{Ne}, evolving to t = {SNAPS[-1]} T_d0 ...")
    sols = solver.evolve(f0, SNAPS[-1], dt, SNAPS)
    for t in SNAPS:
        print(f"  t = {t:5.1f}: surviving fraction = {solver.mass(sols[t]):.4f}")

    # ---------------- plotting (mirrors the panel layout of the paper) ----------------
    plt = setup()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))
    a_cells = np.exp(solver.u)
    e_cells = np.sqrt(np.maximum(solver.v, 1e-12))
    plot_times = [0.1, 0.3, 1.0, 3.0]

    # heatmaps of f(a, e), color normalized per panel, linear a/a0 axis
    e_e = np.sqrt(v_edges)
    for axh, t in zip(axes[0], HEAT_TIMES):
        f_ae = sols[t] * (2 * e_cells)[None, :] / a_cells[:, None]
        pc = axh.pcolormesh(np.exp(x_edges), e_e, (f_ae / f_ae.max()).T,
                            cmap="viridis", rasterized=True, shading="auto")
        axh.set_xlim(0.25, 3)
        axh.set_ylim(0, 1)
        axh.set_xlabel(r"$a/a_0$")
        axh.set_ylabel(r"$e$")
        axh.text(0.97, 0.08, rf"$t/T_{{\mathrm{{d}},0}} = {t:g}$",
                 transform=axh.transAxes, ha="right", color="w")
        fig.colorbar(pc, ax=axh, pad=0.02, label=r"$f/f_\mathrm{max}$")

    # semi-major-axis marginal f(a, t): its area is the surviving fraction
    ax = axes[1, 0]
    for t in plot_times:
        fa = sols[t].sum(axis=1) * solver.dv / a_cells   # f(ln a) -> f(a)
        ax.plot(a_cells, fa, label=rf"${t:g}$")
    ax.set_xlim(0.4, 3)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$a/a_0$")
    ax.set_ylabel(r"$f(a,\, t)$")
    ax.legend(fontsize=8, title=r"$t/T_{\mathrm{d},0}$")

    # eccentricity marginal, renormalized, with f_ss overlay
    ax = axes[1, 1]
    for t in plot_times:
        fv = sols[t].sum(axis=0) * solver.du       # f(eps)
        fe = fv * 2 * e_cells                       # f(e)
        fe /= np.trapz(fe, e_cells)
        ax.plot(e_cells, fe, label=rf"${t:g}$")
    fss = e_cells / (4 + 5 * e_cells**2) ** (36 / 35)
    fss /= np.trapz(fss, e_cells)
    ax.plot(e_cells, fss, "k--", lw=1.2, label=r"$f_\mathrm{ss}(e)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$f(e, t)$ (renormalized)")
    ax.legend(fontsize=8, title=r"$t/T_{\mathrm{d},0}$")

    fig.tight_layout()
    fig.savefig(outdir / "fig3_whitenoise_ae.pdf")
    np.savetxt(outdir / "fig3_fa.dat",
               np.column_stack([a_cells] + [sols[t].sum(axis=1) * solver.dv / a_cells
                                            for t in SNAPS]),
               header="a_over_a0 " + " ".join(f"f_t{t:g}" for t in SNAPS)
               + "   (f per unit a; area = surviving fraction)")
    fe_cols = []
    for t in SNAPS:
        fe = sols[t].sum(axis=0) * solver.du * 2 * e_cells
        fe_cols.append(fe / np.trapz(fe, e_cells))
    np.savetxt(outdir / "fig3_fe.dat",
               np.column_stack([e_cells] + fe_cols + [fss]),
               header="e " + " ".join(f"f_t{t:g}" for t in SNAPS) + " f_ss")
    print(f"wrote {outdir/'fig3_whitenoise_ae.pdf'}")


if __name__ == "__main__":
    main()
