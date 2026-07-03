"""Conservative finite-volume solver for the 2D Fokker--Planck equation

    df/dt = -d_u(B_u f) - d_v(B_v f)
            + (1/2) d_u^2 (D_uu f) + d_u d_v (D_uv f) + (1/2) d_v^2 (D_vv f)

written in flux form df/dt = -d_u J_u - d_v J_v with

    J_u = B_u f - (1/2) d_u (D_uu f) - (1/2) d_v (D_uv f),
    J_v = B_v f - (1/2) d_v (D_vv f) - (1/2) d_u (D_uv f).

Cell-centered grid, fluxes on faces (9-point stencil through the cross term).
Mass conservation is exact by construction for reflecting (zero-flux)
boundaries: every face flux enters the two adjacent cells with opposite
signs. An absorbing boundary is implemented with a zero ghost cell, so the
outgoing face flux removes probability (-> disrupted binaries).

Time stepping: backward Euler (L-stable), with the sparse LU factorization
computed once and reused for every step.

Self-tests (run this module directly):
  * mass conservation to machine precision with all-reflecting boundaries;
  * the analytic stationary states f_0 = sqrt(a) e and
    f_ss = e/(a^4 (4+5e^2)^{36/35}) of the white-noise tidal problem are
    annihilated by the discrete operator to truncation order.
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla


class FP2D:
    def __init__(self, u_edges, v_edges, coeffs, bc=("reflect", "reflect", "reflect", "reflect")):
        """
        u_edges, v_edges : arrays of cell edges (len Nu+1, Nv+1), uniform.
        coeffs : dict with callables Bu, Bv, Duu, Duv, Dvv of (U, V) meshes.
        bc : (u_min, u_max, v_min, v_max), each 'reflect' or 'absorb'.
        """
        self.u = 0.5 * (u_edges[:-1] + u_edges[1:])
        self.v = 0.5 * (v_edges[:-1] + v_edges[1:])
        self.du = u_edges[1] - u_edges[0]
        self.dv = v_edges[1] - v_edges[0]
        self.Nu, self.Nv = len(self.u), len(self.v)
        self.bc = bc
        U, V = np.meshgrid(self.u, self.v, indexing="ij")
        self.Bu = coeffs["Bu"](U, V)
        self.Bv = coeffs["Bv"](U, V)
        self.Duu = coeffs["Duu"](U, V)
        self.Duv = coeffs["Duv"](U, V)
        self.Dvv = coeffs["Dvv"](U, V)
        self.L = self._assemble_fast()
        self._lu = None
        self._dt = None

    # -- vectorized operator assembly (Kronecker form) ---------------------------
    def _face_ops_1d(self, N, h, bc_lo, bc_hi):
        """1D face operators of shape (N+1, N).

        AVG  : ghost-zero average (0.5, 0.5 interior; 0.5 on the single cell
               at absorbing boundary faces; zero row at reflecting faces)
        DIF  : ghost-zero difference /h
        AVGc : cross-term face average (0.5/0.5 interior; weight 1 on the
               single cell at absorbing faces, matching the loop assembly)
        DIV  : divergence, shape (N, N+1): (J_{k+1} - J_k)/h
        """
        AVG = sps.lil_matrix((N + 1, N))
        DIF = sps.lil_matrix((N + 1, N))
        AVGc = sps.lil_matrix((N + 1, N))
        for k in range(1, N):
            AVG[k, k - 1] = 0.5
            AVG[k, k] = 0.5
            AVGc[k, k - 1] = 0.5
            AVGc[k, k] = 0.5
            DIF[k, k - 1] = -1 / h
            DIF[k, k] = 1 / h
        if bc_lo == "absorb":
            AVG[0, 0] = 0.5
            AVGc[0, 0] = 1.0
            DIF[0, 0] = 1 / h
        if bc_hi == "absorb":
            AVG[N, N - 1] = 0.5
            AVGc[N, N - 1] = 1.0
            DIF[N, N - 1] = -1 / h
        rowsD = []
        colsD = []
        valsD = []
        for k in range(N):
            rowsD += [k, k]
            colsD += [k, k + 1]
            valsD += [-1 / h, 1 / h]
        DIV = sps.coo_matrix((valsD, (rowsD, colsD)), shape=(N, N + 1))
        return AVG.tocsr(), DIF.tocsr(), AVGc.tocsr(), DIV.tocsr()

    @staticmethod
    def _grad_1d(N, h):
        """Cell-centered derivative, centered interior / one-sided at edges."""
        Gm = sps.lil_matrix((N, N))
        for k in range(1, N - 1):
            Gm[k, k + 1] = 1 / (2 * h)
            Gm[k, k - 1] = -1 / (2 * h)
        Gm[0, 1] = 1 / h
        Gm[0, 0] = -1 / h
        Gm[N - 1, N - 1] = 1 / h
        Gm[N - 1, N - 2] = -1 / h
        return Gm.tocsr()

    def _assemble_fast(self):
        """Same operator as _assemble (verified identical in the self-test),
        built from Kronecker products of 1D operators."""
        Nu, Nv, du, dv = self.Nu, self.Nv, self.du, self.dv
        Iu = sps.identity(Nu, format="csr")
        Iv = sps.identity(Nv, format="csr")
        AVGu, DIFu, AVGcu, DIVu = self._face_ops_1d(Nu, du, self.bc[0], self.bc[1])
        AVGv, DIFv, AVGcv, DIVv = self._face_ops_1d(Nv, dv, self.bc[2], self.bc[3])
        Gu = self._grad_1d(Nu, du)
        Gv = self._grad_1d(Nv, dv)

        def diag(arr):
            return sps.diags(arr.reshape(-1))

        # J_u = AVG(Bu f) - (1/2) DIF(Duu f) - (1/2) AVGc( d/dv (Duv f) )
        Ju = (sps.kron(AVGu, Iv) @ diag(self.Bu)
              - 0.5 * sps.kron(DIFu, Iv) @ diag(self.Duu)
              - 0.5 * sps.kron(AVGcu, Iv) @ sps.kron(Iu, Gv) @ diag(self.Duv))
        Jv = (sps.kron(Iu, AVGv) @ diag(self.Bv)
              - 0.5 * sps.kron(Iu, DIFv) @ diag(self.Dvv)
              - 0.5 * sps.kron(Iu, AVGcv) @ sps.kron(Gu, Iv) @ diag(self.Duv))
        L = -(sps.kron(DIVu, Iv) @ Ju) - (sps.kron(Iu, DIVv) @ Jv)
        return L.tocsc()

    # -- operator assembly ----------------------------------------------------
    def _assemble(self):
        Nu, Nv, du, dv = self.Nu, self.Nv, self.du, self.dv
        idx = lambda i, j: i * Nv + j
        rows, cols, vals = [], [], []

        def add(n_to, n_from, w):
            rows.append(n_to)
            cols.append(n_from)
            vals.append(w)

        def dv_stencil(i, j):
            """(d/dv (Duv f))_{i,j} as list of (flat_index, weight)."""
            out = []
            if 0 < j < Nv - 1:
                out.append((idx(i, j + 1), self.Duv[i, j + 1] / (2 * dv)))
                out.append((idx(i, j - 1), -self.Duv[i, j - 1] / (2 * dv)))
            elif j == 0:
                out.append((idx(i, j + 1), self.Duv[i, j + 1] / dv))
                out.append((idx(i, j), -self.Duv[i, j] / dv))
            else:
                out.append((idx(i, j), self.Duv[i, j] / dv))
                out.append((idx(i, j - 1), -self.Duv[i, j - 1] / dv))
            return out

        def du_stencil(i, j):
            out = []
            if 0 < i < Nu - 1:
                out.append((idx(i + 1, j), self.Duv[i + 1, j] / (2 * du)))
                out.append((idx(i - 1, j), -self.Duv[i - 1, j] / (2 * du)))
            elif i == 0:
                out.append((idx(i + 1, j), self.Duv[i + 1, j] / du))
                out.append((idx(i, j), -self.Duv[i, j] / du))
            else:
                out.append((idx(i, j), self.Duv[i, j] / du))
                out.append((idx(i - 1, j), -self.Duv[i - 1, j] / du))
            return out

        # ---- u-faces (between (i,j) and (i+1,j)); interior ----
        for i in range(Nu - 1):
            for j in range(Nv):
                face = []  # (flat_index, weight) contributions to J_u
                face.append((idx(i, j), 0.5 * self.Bu[i, j] + self.Duu[i, j] / (2 * du)))
                face.append((idx(i + 1, j), 0.5 * self.Bu[i + 1, j] - self.Duu[i + 1, j] / (2 * du)))
                # -1/2 * face-average of the two adjacent d/dv stencils
                for n_from, wgt in dv_stencil(i, j) + dv_stencil(i + 1, j):
                    face.append((n_from, -0.25 * wgt))
                for n_from, wgt in face:
                    add(idx(i, j), n_from, -wgt / du)
                    add(idx(i + 1, j), n_from, +wgt / du)

        # ---- v-faces (between (i,j) and (i,j+1)); interior ----
        for i in range(Nu):
            for j in range(Nv - 1):
                face = []
                face.append((idx(i, j), 0.5 * self.Bv[i, j] + self.Dvv[i, j] / (2 * dv)))
                face.append((idx(i, j + 1), 0.5 * self.Bv[i, j + 1] - self.Dvv[i, j + 1] / (2 * dv)))
                for n_from, wgt in du_stencil(i, j) + du_stencil(i, j + 1):
                    face.append((n_from, -0.25 * wgt))
                for n_from, wgt in face:
                    add(idx(i, j), n_from, -wgt / dv)
                    add(idx(i, j + 1), n_from, +wgt / dv)

        # ---- boundary faces ----
        # absorbing: ghost cell f = 0 => J = 1/2 B f_in + (D f)_in/(2 h) (outward)
        if self.bc[1] == "absorb":
            i = Nu - 1
            for j in range(Nv):
                face = [(idx(i, j), 0.5 * self.Bu[i, j] + self.Duu[i, j] / (2 * du))]
                for n_from, wgt in dv_stencil(i, j):
                    face.append((n_from, -0.5 * wgt))
                for n_from, wgt in face:
                    add(idx(i, j), n_from, -wgt / du)
        if self.bc[0] == "absorb":
            i = 0
            for j in range(Nv):
                face = [(idx(i, j), 0.5 * self.Bu[i, j] - self.Duu[i, j] / (2 * du))]
                for n_from, wgt in dv_stencil(i, j):
                    face.append((n_from, -0.5 * wgt))
                for n_from, wgt in face:
                    add(idx(i, j), n_from, +wgt / du)
        if self.bc[3] == "absorb":
            j = Nv - 1
            for i in range(Nu):
                face = [(idx(i, j), 0.5 * self.Bv[i, j] + self.Dvv[i, j] / (2 * dv))]
                for n_from, wgt in du_stencil(i, j):
                    face.append((n_from, -0.5 * wgt))
                for n_from, wgt in face:
                    add(idx(i, j), n_from, -wgt / dv)
        if self.bc[2] == "absorb":
            j = 0
            for i in range(Nu):
                face = [(idx(i, j), 0.5 * self.Bv[i, j] - self.Dvv[i, j] / (2 * dv))]
                for n_from, wgt in du_stencil(i, j):
                    face.append((n_from, -0.5 * wgt))
                for n_from, wgt in face:
                    add(idx(i, j), n_from, +wgt / dv)

        N = Nu * Nv
        return sps.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()

    # -- time stepping ---------------------------------------------------------
    def step_factorize(self, dt):
        N = self.Nu * self.Nv
        self._lu = spla.splu((sps.identity(N, format="csc") - dt * self.L).tocsc())
        self._dt = dt

    def evolve(self, f, t_final, dt, snapshots):
        """Backward-Euler evolution; returns {t: f_2d} at the requested times."""
        if self._lu is None or self._dt != dt:
            self.step_factorize(dt)
        f = f.reshape(-1).copy()
        out = {}
        t = 0.0
        snaps = sorted(snapshots)
        k = 0
        nstep = int(np.ceil(snaps[-1] / dt - 1e-9))
        for n in range(nstep):
            f = self._lu.solve(f)
            t = (n + 1) * dt
            while k < len(snaps) and t >= snaps[k] - 1e-9:
                out[snaps[k]] = f.reshape(self.Nu, self.Nv).copy()
                k += 1
        for kk in range(k, len(snaps)):  # flush (dt does not divide t_final)
            out[snaps[kk]] = f.reshape(self.Nu, self.Nv).copy()
        return out

    def mass(self, f):
        return f.sum() * self.du * self.dv


# ------------------------------------------------------------------------------
# self-tests
# ------------------------------------------------------------------------------
def _selftest():
    from coefficients import whitenoise_xe_coeffs

    print("fp2d self-tests (white-noise tidal problem, x = ln a, eps = e^2):")

    # (t0) the vectorized Kronecker assembly equals the reference loop
    # assembly exactly, for every boundary-condition combination
    for bc in [("reflect",) * 4, ("reflect", "absorb", "reflect", "reflect"),
               ("absorb", "absorb", "absorb", "absorb")]:
        s = FP2D(np.linspace(-1, 2, 13), np.linspace(0, 1, 9),
                 whitenoise_xe_coeffs(), bc=bc)
        L_loop = s._assemble()
        diff = np.abs((s.L - L_loop).toarray()).max()
        scale = np.abs(s.L.data).max()
        print(f"  Kron vs loop assembly, bc={bc[1]}/{bc[0]}: rel diff = "
              f"{diff/scale:.2e}  ({'PASS' if diff/scale < 1e-14 else 'FAIL'})")
    Nx, Ne = 120, 80
    xe = np.linspace(-2, 4, Nx + 1)
    ve = np.linspace(0, 1, Ne + 1)
    solver = FP2D(xe, ve, whitenoise_xe_coeffs(),
                  bc=("reflect", "reflect", "reflect", "reflect"))

    # (t1) exact mass conservation with all-reflecting boundaries
    # (column sums vanish face-by-face; the residual is pure roundoff, so
    # normalize by the largest operator entry)
    colsum = np.abs(np.asarray(solver.L.sum(axis=0))).max()
    rel = colsum / np.abs(solver.L.data).max()
    print(f"  max |column sum of L| / max|L| = {rel:.3e}  "
          f"({'PASS' if rel < 1e-12 else 'FAIL'})")

    # (t2) the analytic stationary states are annihilated to truncation order
    X, V = np.meshgrid(solver.u, solver.v, indexing="ij")
    a_grid = np.exp(X)
    e_grid = np.sqrt(np.maximum(V, 1e-12))
    # f in (x, eps) variables: f_xe = f_ae * |da/dx| * |de/deps| = f_ae * a /(2e)
    for name, f_ae in [
        ("f_0 = sqrt(a) e", np.sqrt(a_grid) * e_grid),
        ("f_ss = e/(a^4 (4+5e^2)^{36/35})",
         e_grid / (a_grid**4 * (4 + 5 * e_grid**2) ** (36 / 35))),
    ]:
        f_xe = f_ae * a_grid / (2 * e_grid)
        res = solver.L @ f_xe.reshape(-1)
        # normalize by the operator scale acting on f
        scale = np.abs(solver.L).dot(np.abs(f_xe.reshape(-1))).max()
        rel = np.abs(res).max() / scale
        print(f"  |L f|/scale for {name}: {rel:.2e}  "
              f"({'PASS' if rel < 5e-3 else 'FAIL'})")


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    _selftest()
