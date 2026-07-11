"""Figure 2 (fig:f-adiabatic): relaxation of the eccentricity
distribution to the thermal law f_th = 2e under ADIABATIC tidal
perturbations (sec:adiabatic).

Solves the 1D Fokker--Planck equation eqn:fp-e-adiabatic with the coefficients
B^e (eqn:Ba-adiabatic block) and D^ee, starting from
f(e, 0) ~ delta(e - 0.5), and plots f(e, t) at t/T_d = 0.1, 0.3, 1, 3
together with the thermal distribution (eqn:f-thermal).

Runtime: seconds.  Usage:  python fig2_adiabatic_eccentricity.py [--fast]
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.style import setup
from common.fp1d import solve_adiabatic_e

OUT = pathlib.Path(__file__).resolve().parent / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)

    Nz, dt = (400, 1e-3) if args.fast else (1200, 2e-4)
    times = [0.1, 0.3, 1.0, 3.0]
    e, sols = solve_adiabatic_e(e0=0.5, sigma_e=0.015, Nz=Nz, dt=dt,
                                snapshots=times + [10.0])

    # quantitative gate: at t = 10 T_d the distribution is thermal. The
    # degenerate corner e -> 0 (mobility ~ e^4 in eqn:Je-adiabatic)
    # relaxes arbitrarily slowly, so gate away from it.
    mask = e > 0.2
    dev = np.abs(sols[10.0] - 2 * e)[mask].max()
    print(f"max_(e>0.2) |f(e, 10 T_d) - 2e| = {dev:.2e}  "
          f"({'PASS' if dev < 1e-3 else 'FAIL'} @ 1e-3)")

    plt = setup()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for t in times:
        ax.plot(e, sols[t], label=rf"$t/T_\mathrm{{d}} = {t:g}$")
    ax.plot(e, 2 * e, "k--", lw=1, label=r"$f_\mathrm{th} = 2e$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$f(e, t)$")
    ax.legend()
    fig.savefig(outdir / "fig2_adiabatic_eccentricity.pdf")
    np.savetxt(outdir / "fig2_data.dat",
               np.column_stack([e] + [sols[t] for t in times] + [2 * e]),
               header="e " + " ".join(f"f_t{t:g}" for t in times) + " f_thermal")
    print(f"wrote {outdir/'fig2_adiabatic_eccentricity.pdf'}")


if __name__ == "__main__":
    main()
