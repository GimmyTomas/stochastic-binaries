"""Figure 4 (fig:f-point-mass): numerical solution of the (a, e) sector of
the Fokker--Planck equation for IMPULSIVE encounters with point-mass
perturbers (sec:point-mass), using the closed-form coefficients
eqn:Ba-point-mass .. eqn:Dee-point-mass with the Coulomb logarithm held
FIXED at logLambda = 25, as in the caption.

Time is in units of T_d0 = T_d(a = a0, e = 0.5) computed from eqn:Td*.
Grid: (a/a0, eps = e^2) on [0.1, 16] x [0, 1] (linear x eps); initial
condition, boundary conditions, and panel structure as in Figure 3. The
eccentricity marginal relaxes to the slightly SUB-thermal quasi-stationary
distribution eqn:f-ss-point-mass (overlaid, together with the thermal 2e).

Runtime: ~4 min (default 600x240 grid).  --fast: 300x120, ~30 s.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.style import setup
from common.fp2d import FP2D
from common.coefficients import pointmass_a_eps_coeffs

OUT = pathlib.Path(__file__).resolve().parent / "output"
SNAPS = [0.1, 0.3, 1.0, 3.0, 10.0]
HEAT_TIMES = [0.1, 3.0]
LOGLAM = 25.0


def initial_condition(solver, a0=1.0, e0=0.5, sig_a=0.05, sig_e=0.03):
    A, V = np.meshgrid(solver.u, solver.v, indexing="ij")
    e = np.sqrt(np.maximum(V, 1e-12))
    f_ae = np.exp(-0.5 * (A - a0) ** 2 / sig_a**2 - 0.5 * (e - e0) ** 2 / sig_e**2)
    f_aeps = f_ae / (2 * e)
    f_aeps /= f_aeps.sum() * solver.du * solver.dv
    return f_aeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)

    Na, Ne, dt = (300, 120, 2.4e-2) if args.fast else (600, 240, 1.2e-2)
    a_edges = np.linspace(0.1, 16.0, Na + 1)
    v_edges = np.linspace(0, 1, Ne + 1)
    solver = FP2D(a_edges, v_edges, pointmass_a_eps_coeffs(logLam=LOGLAM, e0=0.5),
                  bc=("reflect", "absorb", "reflect", "reflect"))
    f0 = initial_condition(solver)
    print(f"grid {Na}x{Ne}, logLambda = {LOGLAM}, evolving to t = {SNAPS[-1]} T_d0 ...")
    sols = solver.evolve(f0, SNAPS[-1], dt, SNAPS)
    for t in SNAPS:
        print(f"  t = {t:5.1f}: surviving fraction = {solver.mass(sols[t]):.4f}")

    # ---------------- plotting (mirrors the panel layout of the paper) ----------------
    plt = setup()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))
    a_cells = solver.u
    e_cells = np.sqrt(np.maximum(solver.v, 1e-12))
    e_e = np.sqrt(v_edges)
    plot_times = [0.3, 1.0, 3.0, 10.0]

    for axh, t in zip(axes[0], HEAT_TIMES):
        f_ae = sols[t] * (2 * e_cells)[None, :]
        pc = axh.pcolormesh(a_edges, e_e, (f_ae / f_ae.max()).T,
                            cmap="viridis", rasterized=True, shading="auto")
        axh.set_xlim(0.1, 4)
        axh.set_ylim(0, 1)
        axh.set_xlabel(r"$a/a_0$")
        axh.set_ylabel(r"$e$")
        axh.text(0.97, 0.08, rf"$t/T_{{\mathrm{{d}},0}} = {t:g}$",
                 transform=axh.transAxes, ha="right", color="w")
        fig.colorbar(pc, ax=axh, pad=0.02, label=r"$f/f_\mathrm{max}$")

    ax = axes[1, 0]
    for t in plot_times:
        fa = sols[t].sum(axis=1) * solver.dv
        ax.plot(a_cells, fa, label=rf"${t:g}$")
    ax.set_xlim(0.1, 4)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$a/a_0$")
    ax.set_ylabel(r"$f(a, t)$")
    ax.legend(fontsize=8, title=r"$t/T_{\mathrm{d},0}$")

    ax = axes[1, 1]
    for t in plot_times:
        fv = sols[t].sum(axis=0) * solver.du
        fe = fv * 2 * e_cells
        fe /= np.trapz(fe, e_cells)
        ax.plot(e_cells, fe, label=rf"${t:g}$")
    w = np.sqrt(1 - e_cells**2)
    chi = 4 * w - 2 * np.log(1 + w)
    fss = 2 * e_cells * (1 + chi / LOGLAM)
    fss /= np.trapz(fss, e_cells)
    fth = 2 * e_cells / np.trapz(2 * e_cells, e_cells)
    ax.plot(e_cells, fss, "k--", lw=1.2, label=r"$f_\mathrm{ss}(e)$")
    ax.plot(e_cells, fth, "k:", lw=1.2, label=r"$f_\mathrm{th} = 2e$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$f(e, t)$ (renormalized)")
    ax.legend(fontsize=8, title=r"$t/T_{\mathrm{d},0}$")

    fig.tight_layout()
    fig.savefig(outdir / "fig4_pointmass_ae.pdf")
    fe_cols = []
    for t in SNAPS:
        fe = sols[t].sum(axis=0) * solver.du * 2 * e_cells
        fe_cols.append(fe / np.trapz(fe, e_cells))
    np.savetxt(outdir / "fig4_fe.dat",
               np.column_stack([e_cells] + fe_cols + [fss, fth]),
               header="e " + " ".join(f"f_t{t:g}" for t in SNAPS) + " f_ss f_th")
    print(f"wrote {outdir/'fig4_pointmass_ae.pdf'}")


if __name__ == "__main__":
    main()
