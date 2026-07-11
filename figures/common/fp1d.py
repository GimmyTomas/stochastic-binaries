"""1D Fokker--Planck solver for the adiabatic tidal eccentricity problem
(eqn:fp-e-adiabatic).

With g = f/(2e) and z = e^2 the equation becomes the manifestly regular
divergence form

    dg/dt = (5/6) d/dz [ z^2 (1 - z) dg/dz ] ,        t in units of T_d,

whose stationary solution g = const is the thermal distribution f = 2e
(eqn:f-thermal). Cell-centered conservative finite volumes with zero-flux
boundaries (both e = 0 and e = 1 are inaccessible: the flux z^2(1-z) dg/dz
vanishes there for regular g), backward-Euler time stepping.
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


def solve_adiabatic_e(e0=0.5, sigma_e=0.015, Nz=1200, dt=2e-4, snapshots=(0.1, 0.3, 1.0, 3.0)):
    """Return (e_centers, {t: f(e)}) for the initial condition ~ delta(e - e0)."""
    z_edges = np.linspace(0.0, 1.0, Nz + 1)
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = z_edges[1] - z_edges[0]
    mob = (5.0 / 6.0) * z_edges[1:-1] ** 2 * (1 - z_edges[1:-1])  # interior faces

    # d g_i/dt = (1/dz) [ F_{i+1/2} - F_{i-1/2} ],  F = mob * (g_{i+1}-g_i)/dz
    main = np.zeros(Nz)
    upper = np.zeros(Nz - 1)
    lower = np.zeros(Nz - 1)
    main[:-1] -= mob / dz**2
    upper[:] += mob / dz**2
    main[1:] -= mob / dz**2
    lower[:] += mob / dz**2
    L = sps.diags([lower, main, upper], [-1, 0, 1], format="csc")

    e = np.sqrt(z)
    f_e = np.exp(-0.5 * (e - e0) ** 2 / sigma_e**2)
    g = f_e / (2 * e)
    # normalize: Int f de = Int g dz = 1
    g /= g.sum() * dz

    lu = spla.splu((sps.identity(Nz, format="csc") - dt * L).tocsc())
    out = {}
    snaps = sorted(snapshots)
    nstep = int(round(snaps[-1] / dt))
    k = 0
    for n in range(nstep):
        g = lu.solve(g)
        t = (n + 1) * dt
        while k < len(snaps) and t >= snaps[k] - 1e-12:
            out[snaps[k]] = 2 * e * g.copy()
            k += 1
    return e, out
