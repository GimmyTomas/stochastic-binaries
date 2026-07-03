"""Figure 6 (fig:fp_vs_mc): Fokker--Planck vs brute-force Monte Carlo for
binaries undergoing impulsive encounters with uniform-density spherical
perturbers (sec:Gai; setup of Ramirez et al. 2022).

Common parameters (caption): sigma = 200 km/s, rho_bar = 0.0104 Msun/pc^3,
m_* = 10^3 Msun, e0 = 0.5, total binary mass m = 1 Msun, evolution time
10 Gyr. Cases: R = 0.1 pc with a0 = 0.01, 0.05 pc (left panel) and R = 1 pc
with a0 = 0.05, 0.10 pc (right panel).

* MONTE CARLO: each binary receives a Poisson number of independent
  impulsive kicks. Per encounter: flux-weighted Maxwellian speed (chi_4),
  isotropic geometry, impact parameter uniform in area out to a fixed
  p_max = max(5 a0, 3 R, 0.3 pc, [distant-kick criterion]), exact two-body
  impulses on both stars with the homogeneous-sphere form factor I(b)
  (Table I), energy/angular-momentum update, disruption when E >= 0.
* FOKKER--PLANCK: the general coefficients eqn:Ba-impulsive ..
  eqn:Dee-impulsive orbit-averaged over tabulated Q_r(r), Q_t(r) (built by
  common/kick_covariance.py from the draft's Fourier route, gated against an
  independent real-space quadrature), solved with the conservative solver of
  common/fp2d.py in (x = ln(a/a0), eps = e^2); plus the tidal-limit FP
  (Q_r = 3 A r^2, Q_t = A r^2).

The (R = 0.1, a0 = 0.05) case violates the shot-noise condition
(eqn:poisson-large): the Fokker--Planck description breaks down there and
over-ionizes -- the visible MC/FP disagreement in the left panel is physics,
not a bug (see the discussion in the draft).

Runtime: ~10-20 min default (1e6 binaries/case).  --fast: 1e5, ~2 min.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.style import setup, COLORS
from common.fp2d import FP2D
from common.coefficients import generic_q_ae, _ito_ae, selfcheck_generic_q
from common.kick_covariance import SphereKickCovariance, _I_of_b

OUT = pathlib.Path(__file__).resolve().parent / "output"

# units: pc, km/s, Msun; time unit = pc/(km/s) = 0.9778 Myr
G = 4.300917270e-3          # pc (km/s)^2 / Msun
GYR = 1022.71               # time units per Gyr
SIGMA = 200.0
RHO = 0.0104
MSTAR = 1000.0
NDEN = RHO / MSTAR
MBIN = 1.0
T_TOTAL = 10 * GYR
E0 = 0.5
VMEAN = np.sqrt(8 / np.pi) * SIGMA

CASES = [  # (label, R, a0, panel)
    ("R01_a001", 0.1, 0.01, 0),
    ("R01_a005", 0.1, 0.05, 0),
    ("R10_a005", 1.0, 0.05, 1),
    ("R10_a010", 1.0, 0.10, 1),
]

# reference values from the C++ engine used for the paper (seed-dependent MC;
# agreement is expected within binomial statistics)
REF_SURV_MC = {"R01_a001": 0.999978, "R01_a005": 0.596456,
               "R10_a005": 0.999999, "R10_a010": 0.998692}
REF_SURV_FP = {"R01_a001": 1.000000, "R01_a005": 0.317755,
               "R10_a005": 1.000000, "R10_a010": 0.998321}


# ---------------------------------------------------------------------------
# Monte Carlo engine (vectorized NumPy)
# ---------------------------------------------------------------------------
def kepler_E(M, e, iters=16):
    E = M + e * np.sin(M)
    for _ in range(iters):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def run_mc(R, a0, nmc, rng, chunk=250_000):
    pmax = max(5 * a0, 3 * R, 0.3)
    Eb0 = G * MBIN / (2 * a0)
    num = NDEN * T_TOTAL * np.pi * (2 * G * MSTAR * a0) ** 2 / VMEAN
    pmax = max(pmax, np.sqrt(num / (0.01 * Eb0)))
    Nmean = NDEN * np.pi * pmax**2 * VMEAN * T_TOTAL
    print(f"  MC {R=} {a0=}: pmax = {pmax:.3f} pc, <N_enc> = {Nmean:.1f}")

    finals = []
    ndis = 0
    done = 0
    while done < nmc:
        nb = min(chunk, nmc - done)
        done += nb
        a = np.full(nb, a0)
        e = np.full(nb, E0)
        alive = np.ones(nb, bool)
        Ne = rng.poisson(Nmean, nb)
        kmax = Ne.max()
        for k in range(kmax):
            act = alive & (k < Ne)
            if not act.any():
                break
            idx = np.nonzero(act)[0]
            n_act = idx.size
            aa, ee = a[idx], e[idx]
            # orbital phase and state (orbital frame: orbit in its own plane)
            M = rng.uniform(0, 2 * np.pi, n_act)
            E = kepler_E(M, ee)
            cE, sE = np.cos(E), np.sin(E)
            w = np.sqrt(np.maximum(1 - ee**2, 0))
            r_orb = np.stack([aa * (cE - ee), aa * w * sE, np.zeros(n_act)], 1)
            nmot = np.sqrt(G * MBIN / aa**3)
            den = 1 - ee * cE
            v_orb = np.stack([-aa * nmot * sE / den,
                              aa * nmot * w * cE / den,
                              np.zeros(n_act)], 1)
            # perturber: speed (flux-weighted Maxwellian = chi_4), direction,
            # impact parameter in the plane orthogonal to Vhat
            V = SIGMA * np.sqrt(-2 * np.log(rng.uniform(1e-300, 1, n_act) *
                                            rng.uniform(1e-300, 1, n_act)))
            ct = rng.uniform(-1, 1, n_act)
            st = np.sqrt(1 - ct**2)
            ph = rng.uniform(0, 2 * np.pi, n_act)
            Vhat = np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)
            # orthonormal basis of the b-plane
            helper = np.zeros((n_act, 3))
            helper[:, 0] = 1.0
            swap = np.abs(Vhat[:, 0]) > 0.9
            helper[swap] = [0, 1, 0]
            e1 = np.cross(Vhat, helper)
            e1 /= np.linalg.norm(e1, axis=1)[:, None]
            e2 = np.cross(Vhat, e1)
            p = pmax * np.sqrt(rng.uniform(0, 1, n_act))
            psi = rng.uniform(0, 2 * np.pi, n_act)
            b_com = p[:, None] * (np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2)
            # projected offsets of the two (equal-mass) stars from the line
            rp = r_orb - (r_orb * Vhat).sum(1)[:, None] * Vhat
            b1 = b_com + 0.5 * rp
            b2 = b_com - 0.5 * rp
            n1 = np.linalg.norm(b1, axis=1)
            n2 = np.linalg.norm(b2, axis=1)
            K = -2 * G * MSTAR / V
            f1 = K * _I_of_b(n1 / R) / np.maximum(n1, 1e-25) ** 2
            f2 = K * _I_of_b(n2 / R) / np.maximum(n2, 1e-25) ** 2
            dv = f1[:, None] * b1 - f2[:, None] * b2   # relative kick (star1 - star2)
            # energy / angular momentum update (per reduced mass)
            rr = np.linalg.norm(r_orb, axis=1)
            Enew = 0.5 * ((v_orb + dv) ** 2).sum(1) - G * MBIN / rr
            Jnew = np.cross(r_orb, v_orb + dv)
            J2 = (Jnew**2).sum(1)
            bound = Enew < 0
            anew = np.where(bound, -G * MBIN / (2 * np.where(bound, Enew, -1.0)), np.inf)
            e2new = 1 - J2 / (G * MBIN * np.where(bound, anew, 1.0))
            okay = bound & (e2new < 1) & (anew > 0) & (anew < 1e3)
            a[idx] = np.where(okay, anew, a[idx])
            e[idx] = np.where(okay, np.sqrt(np.clip(e2new, 0, 1 - 1e-12)), e[idx])
            alive[idx] &= okay
        ndis += int((~alive).sum())
        finals.append(a[alive].copy())
    finals = np.concatenate(finals) if finals else np.array([])
    return finals, ndis, nmc


# ---------------------------------------------------------------------------
# Fokker--Planck solves
# ---------------------------------------------------------------------------
def run_fp(cov, a0, tidal=False, Nx=600, Ne=120, nsteps=2000):
    x_edges = np.linspace(-2.5, 5.5, Nx + 1)
    v_edges = np.linspace(0, 1, Ne + 1)
    x_cells = 0.5 * (x_edges[:-1] + x_edges[1:])
    v_cells = 0.5 * (v_edges[:-1] + v_edges[1:])
    Xg, Vg = np.meshgrid(x_cells, v_cells, indexing="ij")
    a_g = a0 * np.exp(Xg)
    e_g = np.sqrt(np.maximum(Vg, 1e-12))
    if tidal:
        A = cov.tidal_amplitude()
        Qr_f = lambda r: 3 * A * r**2
        Qt_f = lambda r: A * r**2
    else:
        # tabulate Q_r, Q_t on a 1D log grid once, then evaluate by spline
        # (Q > 0 and log-smooth, so a log-log cubic spline is accurate)
        from scipy.interpolate import CubicSpline
        r_tab = np.geomspace(1e-5 * a0, 4 * a0 * np.exp(5.5), 500)
        Qr_tab, Qt_tab = cov.Q_rt(r_tab)
        spl_r = CubicSpline(np.log(r_tab), np.log(Qr_tab))
        spl_t = CubicSpline(np.log(r_tab), np.log(Qt_tab))

        def Qr_f(r):
            return np.exp(spl_r(np.log(np.clip(r, r_tab[0], r_tab[-1]))))

        def Qt_f(r):
            return np.exp(spl_t(np.log(np.clip(r, r_tab[0], r_tab[-1]))))
    Ba, Be, Daa, Dae, Dee = generic_q_ae(a_g, e_g, Qr_f, Qt_f, Gm=G * MBIN, NE=96)
    Bu, Bv, Duu, Duv, Dvv = _ito_ae(Ba, Be, Daa, Dae, Dee, a_g, e_g, log_a=True)
    coeffs = {"Bu": lambda X, V_: Bu, "Bv": lambda X, V_: Bv,
              "Duu": lambda X, V_: Duu, "Duv": lambda X, V_: Duv,
              "Dvv": lambda X, V_: Dvv}
    solver = FP2D(x_edges, v_edges, coeffs,
                  bc=("reflect", "absorb", "reflect", "reflect"))
    f0 = np.exp(-0.5 * (a0 * np.exp(Xg) - a0) ** 2 / (0.05 * a0) ** 2
                - 0.5 * (e_g - E0) ** 2 / 0.03**2)
    f0 = f0 * a0 * np.exp(Xg) / (2 * e_g)
    f0 /= f0.sum() * solver.du * solver.dv
    dt = T_TOTAL / nsteps
    sols = solver.evolve(f0, T_TOTAL, dt, [T_TOTAL])
    f_final = sols[T_TOTAL]
    surv = solver.mass(f_final)
    fx = f_final.sum(axis=1) * solver.dv          # f(x); area = survival
    return np.exp(x_cells), fx, surv


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--nmc", type=int, default=None)
    ap.add_argument("--replot", action="store_true",
                    help="reuse cached results (output/fig6_results.npz)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)
    nmc = args.nmc or (100_000 if args.fast else 1_000_000)
    cache_file = outdir / "fig6_results.npz"

    results = {}
    if args.replot and cache_file.exists():
        dat = np.load(cache_file, allow_pickle=True)
        results = dat["results"].item()
        print(f"loaded cached results from {cache_file}")
    else:
        print("validation gate: tidal-limit Q reproduces closed-form coefficients...")
        selfcheck_generic_q()
        print("  PASS")

        rng = np.random.default_rng(20260703)
        for label, R, a0, panel in CASES:
            t0 = time.time()
            cov = SphereKickCovariance(R, G, MSTAR, NDEN, SIGMA,
                                       gate=(label == CASES[0][0]))
            finals, ndis, ntot = run_mc(R, a0, nmc, rng)
            surv_mc = 1 - ndis / ntot
            u_fp, fx_full, surv_full = run_fp(cov, a0, tidal=False)
            _, fx_tidal, surv_tidal = run_fp(cov, a0, tidal=True)
            results[label] = dict(R=R, a0=a0, panel=panel, finals=finals,
                                  surv_mc=surv_mc, u_fp=u_fp,
                                  fx_full=fx_full, surv_full=surv_full,
                                  fx_tidal=fx_tidal, surv_tidal=surv_tidal)
            print(f"  {label}: surv MC = {surv_mc:.6f} (ref {REF_SURV_MC[label]:.6f}), "
                  f"FP full = {surv_full:.6f} (ref {REF_SURV_FP[label]:.6f}), "
                  f"FP tidal = {surv_tidal:.6f}  [{time.time()-t0:.0f}s]")
            # statistical comparison with the reference MC (binomial; both
            # runs carry their own noise)
            pref = REF_SURV_MC[label]
            sig = np.sqrt(max(pref * (1 - pref), 1e-7 / 4) / ntot +
                          pref * (1 - pref) / 1_000_000 + 1e-12)
            dev = abs(surv_mc - pref) / sig
            print(f"      MC vs reference: {dev:.1f} sigma "
                  f"({'OK' if dev < 4 else 'CHECK'})")
        np.savez_compressed(cache_file, results=np.array(results, dtype=object))

    # ---------------- plotting (mirrors the paper's panels) ----------------
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    styles = {}
    for i, (label, R, a0, panel) in enumerate(CASES):
        styles[label] = COLORS[i % 2 if panel == 0 else 2 + i % 2]
    for label, R, a0, panel in CASES:
        res = results[label]
        ax = axes[panel]
        col = styles[label]
        # MC histogram (log-spaced bins; NOT renormalized: area = survival)
        bins = np.geomspace(max(res["finals"].min(), 1e-3 * a0) if res["finals"].size
                            else a0 / 2, 300 * a0, 141)
        hist, edges = np.histogram(res["finals"] / a0, bins=bins / a0)
        cent = np.sqrt(edges[:-1] * edges[1:])
        pdf = hist / np.diff(edges) / len(res["finals"]) * res["surv_mc"]
        ax.plot(cent, pdf, color=col, lw=1.6,
                label=rf"MC, $a_0 = {a0:g}\,$pc")
        # FP full (dashed); the tidal-limit FP is written to the data file
        ax.plot(res["u_fp"], res["fx_full"] / res["u_fp"], "--", color=col, lw=1.3,
                label=rf"FP, $a_0 = {a0:g}\,$pc")
    for panel, R in ((0, 0.1), (1, 1.0)):
        ax = axes[panel]
        ax.set_yscale("log")
        ax.set_xlim(0.3, 2.9)
        ax.set_ylim(1e-2, 12)
        ax.set_xlabel(r"$a/a_0$")
        ax.set_ylabel(r"$f(a)$ after $10\,$Gyr")
        ax.text(0.03, 0.06, rf"$R = {R}\,$pc", transform=ax.transAxes,
                bbox=dict(boxstyle="square", fc="w", ec="0.6", lw=0.6))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig6_fp_vs_mc.pdf")
    for label, R, a0, panel in CASES:
        res = results[label]
        np.savetxt(outdir / f"fig6_fp_{label}.dat",
                   np.column_stack([res["u_fp"], res["fx_full"] / res["u_fp"],
                                    res["fx_tidal"] / res["u_fp"]]),
                   header="a_over_a0 f_fp_full f_fp_tidal")

    with open(outdir / "fig6_survival.dat", "w") as fh:
        fh.write("# case surv_mc surv_fp_full surv_fp_tidal (10 Gyr)\n")
        for label in results:
            r = results[label]
            fh.write(f"{label} {r['surv_mc']:.6f} {r['surv_full']:.6f} "
                     f"{r['surv_tidal']:.6f}\n")
    print(f"wrote {outdir/'fig6_fp_vs_mc.pdf'}")


if __name__ == "__main__":
    main()
