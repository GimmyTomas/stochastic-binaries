"""Figure 7 (fig:white-noise-numerics, Appendix sec:numerical-check):
non-perturbative Monte-Carlo verification of the white-noise tidal
Fokker--Planck coefficients eqn:Ba-white-noise .. eqn:Dee-white-noise.

Method (as described in the appendix):
* The stochastic tidal tensor is built as T_ij(t) = zeta(t) delta_ij +
  sum_A xi_A(t) e^A_ij with an orthonormal basis of five symmetric traceless
  matrices; zeta and xi_A are independent Ornstein--Uhlenbeck processes with
  autocorrelation exp(-|tau|/tau_c) and variances (exact update rule)
      sigma_zeta^2 = Gm/(18 a^3 T_d tau_c),
      sigma_xi^2   = Gm/(15 a^3 T_d tau_c),
  which reproduces the correlator eqn:Td-tidal exactly.
* An ensemble of binaries (e0 = 0.3, mean anomalies uniform) is integrated
  with a leapfrog (kick-drift-kick) scheme for 3 orbital periods, each binary
  under an independent realization -- and its ANTITHETIC partner
  (-zeta, -xi_A), which cancels the first-order noise contribution to the
  drift (eqn discussion in the appendix).
* B^a, B^e are measured from antithetic-pair means (errors: SEM);
  D^aa, D^ae, D^ee from raw increment covariances (eqn:Dmunu-numerical;
  errors: bootstrap over binaries).

Panels (as in the paper): measured/predicted ratio for
[B^a, B^e, D^aa, D^ae, D^ee], (left) T/T_d in {1e-4, 1e-3, 1e-2} at fixed
tau_c/T = 5e-3; (right) tau_c/T in {5e-3, 1e-2, 2e-2} at fixed T/T_d = 1e-5.

Runtime: ~20-40 min default (2e5 antithetic pairs per point; cached in
output/fig7_cache.json).  --fast: 2e4 pairs, ~3 min.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.style import setup, COLORS

OUT = pathlib.Path(__file__).resolve().parent / "output"

GM = 1.0
A0 = 1.0
E0 = 0.3
N_ORBITS = 3
COEFF_NAMES = [r"$B^a$", r"$B^e$", r"$D^{aa}$", r"$D^{ae}$", r"$D^{ee}$"]

# orthonormal basis of symmetric traceless 3x3 matrices (Frobenius norm 1)
S2 = 1 / np.sqrt(2)
S6 = 1 / np.sqrt(6)
TRACELESS_BASIS = np.array([
    [[S2, 0, 0], [0, -S2, 0], [0, 0, 0]],
    [[S6, 0, 0], [0, S6, 0], [0, 0, -2 * S6]],
    [[0, S2, 0], [S2, 0, 0], [0, 0, 0]],
    [[0, 0, S2], [0, 0, 0], [S2, 0, 0]],
    [[0, 0, 0], [0, 0, S2], [0, S2, 0]],
])


def predicted(a, e, invTd):
    """eqn:Ba-white-noise .. eqn:Dee-white-noise."""
    return np.array([
        a * (18 + 19 * e**2) / 30 * invTd,
        (28 - 51 * e**2 - 103 * e**4) / (240 * e) * invTd,
        2 * a**2 * (2 + e**2) / 15 * invTd,
        -2 * a * e * (1 - e**2) / 15 * invTd,
        7 * (1 - e**2) * (4 + 5 * e**2) / 120 * invTd,
    ])


def kepler_E(M, e, iters=14):
    E = M + e * np.sin(M)
    for _ in range(iters):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def elements_from_state(r, v):
    """(a, e) from cartesian state (per reduced mass), vectorized."""
    rr = np.linalg.norm(r, axis=1)
    v2 = (v**2).sum(1)
    En = 0.5 * v2 - GM / rr
    a = -GM / (2 * En)
    J = np.cross(r, v)
    e2 = 1 - (J**2).sum(1) / (GM * a)
    return a, np.sqrt(np.clip(e2, 0, None))


def run_point(T_over_Td, tauc_over_T, n_pairs, seed, n_boot=200):
    """Measure the 5 coefficient ratios (value, error) at one parameter point."""
    rng = np.random.default_rng(seed)
    T_orb = 2 * np.pi * np.sqrt(A0**3 / GM)
    tau_c = tauc_over_T * T_orb
    invTd = T_over_Td / T_orb          # 1/T_d
    sig_zeta = np.sqrt(GM / (18 * A0**3) * invTd / tau_c)
    sig_xi = np.sqrt(GM / (15 * A0**3) * invTd / tau_c)
    dt = tau_c / 8
    n_steps = int(np.ceil(N_ORBITS * T_orb / dt))
    dt = N_ORBITS * T_orb / n_steps
    decay = np.exp(-dt / tau_c)
    kick_noise = np.sqrt(1 - decay**2)

    N = 2 * n_pairs  # antithetic partners interleaved [x, -x]
    # initial conditions: e0 = 0.3, uniform mean anomaly (same for partners)
    M0 = np.repeat(rng.uniform(0, 2 * np.pi, n_pairs), 2)
    E = kepler_E(M0, E0)
    w = np.sqrt(1 - E0**2)
    r = np.stack([A0 * (np.cos(E) - E0), A0 * w * np.sin(E), np.zeros(N)], 1)
    den = 1 - E0 * np.cos(E)
    nmot = np.sqrt(GM / A0**3)
    v = np.stack([-A0 * nmot * np.sin(E) / den,
                  A0 * nmot * w * np.cos(E) / den, np.zeros(N)], 1)

    # OU state: 6 processes per pair; partner value = -value (antithetic)
    ou = np.empty((N, 6))
    ou_half = rng.standard_normal((n_pairs, 6))
    ou[0::2] = ou_half
    ou[1::2] = -ou_half

    def tidal_acc(rv, ouv):
        # T_ij = sig_zeta*zeta delta_ij + sig_xi * sum_A xi_A e^A_ij
        Tm = (sig_xi * np.einsum("nA,Aij->nij", ouv[:, :5], TRACELESS_BASIS)
              + sig_zeta * ouv[:, 5, None, None] * np.eye(3))
        return -np.einsum("nij,nj->ni", Tm, rv)

    # leapfrog (kick-drift-kick), OU refreshed each step (exact update)
    acc_g = -GM * r / np.linalg.norm(r, axis=1)[:, None] ** 3
    acc_t = tidal_acc(r, ou)
    for _ in range(n_steps):
        v += 0.5 * dt * (acc_g + acc_t)
        r += dt * v
        z = rng.standard_normal((n_pairs, 6))
        zz = np.empty((N, 6))
        zz[0::2] = z
        zz[1::2] = -z
        ou = decay * ou + kick_noise * zz
        acc_g = -GM * r / np.linalg.norm(r, axis=1)[:, None] ** 3
        acc_t = tidal_acc(r, ou)
        v += 0.5 * dt * (acc_g + acc_t)

    a_f, e_f = elements_from_state(r, v)
    Dt = N_ORBITS * T_orb
    da = a_f - A0
    de = e_f - E0
    pred = predicted(A0, E0, invTd)

    # drifts from antithetic-pair means
    da_pair = 0.5 * (da[0::2] + da[1::2])
    de_pair = 0.5 * (de[0::2] + de[1::2])
    Ba = da_pair.mean() / Dt
    Be = de_pair.mean() / Dt
    Ba_err = da_pair.std(ddof=1) / np.sqrt(n_pairs) / Dt
    Be_err = de_pair.std(ddof=1) / np.sqrt(n_pairs) / Dt

    # diffusion from raw covariances (eqn:Dmunu-numerical)
    def dcoef(x, y):
        return (np.mean(x * y) - x.mean() * y.mean()) / Dt

    Daa = dcoef(da, da)
    Dae = dcoef(da, de)
    Dee = dcoef(de, de)
    boot = np.empty((n_boot, 3))
    for b in range(n_boot):
        i = rng.integers(0, N, N)
        boot[b] = [dcoef(da[i], da[i]), dcoef(da[i], de[i]), dcoef(de[i], de[i])]
    Derr = boot.std(axis=0, ddof=1)

    vals = np.array([Ba, Be, Daa, Dae, Dee])
    errs = np.array([Ba_err, Be_err, Derr[0], Derr[1], Derr[2]])
    return vals / pred, errs / np.abs(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--pairs", type=int, default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(exist_ok=True)
    n_pairs = args.pairs or (20_000 if args.fast else 200_000)

    series = {
        "weak": {"tauc_over_T": 5e-3,
                 "points": [("1e-4", 1e-4), ("1e-3", 1e-3), ("1e-2", 1e-2)]},
        "white": {"T_over_Td": 1e-5,
                  "points": [("5e-3", 5e-3), ("1e-2", 1e-2), ("2e-2", 2e-2)]},
    }
    cache_path = outdir / "fig7_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    results = {}
    for sname, spec in series.items():
        for label, val in spec["points"]:
            if sname == "weak":
                T_over_Td, tauc_over_T = val, spec["tauc_over_T"]
            else:
                T_over_Td, tauc_over_T = spec["T_over_Td"], val
            key = hashlib.md5(
                f"{T_over_Td}_{tauc_over_T}_{n_pairs}_{E0}_{N_ORBITS}".encode()
            ).hexdigest()[:16]
            if key in cache:
                results[(sname, label)] = cache[key]
                print(f"  [{sname} {label}] cached")
                continue
            t0 = time.time()
            ratios, errs = run_point(T_over_Td, tauc_over_T, n_pairs,
                                     seed=abs(hash((sname, label))) % 2**31)
            results[(sname, label)] = [list(ratios), list(errs)]
            cache[key] = results[(sname, label)]
            cache_path.write_text(json.dumps(cache, indent=1))
            print(f"  [{sname} {label}] ratios = "
                  + " ".join(f"{x:.4f}" for x in ratios)
                  + f"  ({time.time()-t0:.0f}s)")

    # gate: at the most-converged points all ratios within the 2% band
    for sname, label in (("weak", "1e-4"), ("white", "5e-3")):
        ratios, errs = results[(sname, label)]
        dev = np.abs(np.array(ratios) - 1)
        ok = (dev < np.maximum(0.02, 3 * np.array(errs))).all()
        print(f"gate [{sname} {label}]: max|ratio-1| = {dev.max():.3f} "
              f"({'PASS' if ok else 'FAIL'} @ 2% or 3 sigma)")

    # ---------------- plotting ----------------
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    xpos = np.arange(5)
    for ax, sname, title, leg in (
        (axes[0], "weak", r"$\tau_\mathrm{c}/T = 5\times10^{-3}$", r"$T/T_\mathrm{d}$"),
        (axes[1], "white", r"$T/T_\mathrm{d} = 10^{-5}$", r"$\tau_\mathrm{c}/T$"),
    ):
        ax.axhspan(0.98, 1.02, color="0.85", zorder=0)
        ax.axhline(1, color="k", lw=0.7, zorder=1)
        for k, (label, _) in enumerate(series[sname]["points"]):
            ratios, errs = results[(sname, label)]
            ax.errorbar(xpos + (k - 1) * 0.18, ratios, yerr=errs, fmt="o",
                        ms=4, color=COLORS[k], label=rf"${label}$", capsize=2)
        ax.set_xticks(xpos)
        ax.set_xticklabels(COEFF_NAMES)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, title=leg)
    axes[0].set_ylabel("measured / predicted")
    fig.tight_layout()
    fig.savefig(outdir / "fig7_coefficient_convergence.pdf")
    print(f"wrote {outdir/'fig7_coefficient_convergence.pdf'}")


if __name__ == "__main__":
    main()
