#!/usr/bin/env python3
r"""
Trajectory-level checks of the ingredients of the b90-exactness proof
(regularization-exactness.tex, Lemmas "Incoming-map regularity" and "Kick fidelity",
plus the pinned-vs-recoil Remark).

The companion scripts test the CLOSED FORMS (exact_deflection_check.py) and the
INTEGRATED covariance (three_body_check.py).  This script tests the proof's
per-trajectory ingredients directly:

  1. MODEL MATTERS (Remark "pinned vs recoil"): freezing the binary must mean
     "internal binary force off, bodies recoil".  If the bodies are PINNED, the
     single-body encounter regularizes at Gm_i/V^2; with recoil, at G(m_i+m*)/V^2.
     We integrate both models and extract the cutoff c from
     |dv1|^2 = 4 G^2 m*^2 / [V^2 (beta^2 + c^2)].

  2. DEFLECTION MAP (Lemma "Incoming-map regularity"): for the genuine 3-body
     encounter, the map b -> beta1* (VECTOR osculating impact parameter of the
     m*-body1 relative orbit at closest approach) satisfies
        |delta| = |beta1* - (b - s1_proj)| = O(b90)   (r-independent monopole shift),
        det(d beta1*/d b) = 1 + O(eta)                (area-preserving relabeling),
        V1* = V [1 + O(eta)].
     The VECTOR map must be used: the O(b90) translation also rotates beta1* by an
     O(b90/beta1) angle that a scalar |beta1| diagnostic would misread as a Jacobian
     distortion.  The 2x2 Jacobian is computed by central finite differences.

  3. KICK FIDELITY (Lemma "Kick fidelity"): the actual kick to body 1 equals the
     isolated two-body kick at the osculating elements,
        |dv1|^2 = (4 G^2 m*^2 / V1^2) / (beta1*^2 + b90(V1*)^2) * [1 + eps],
     with eps -> 0 as eta -> 0.  We also show that using the asymptotic V instead of
     V1* (i.e. dropping the (V/V1*)^2 prefactor and the b90(V1*) shift) fits WORSE.

G = m* = 1 throughout.
"""
import sys, time
import numpy as np
from scipy.integrate import solve_ivp

PASS = lambda ok: "PASS" if ok else "**FAIL**"
G = 1.0; mstar = 1.0

# --------------------------------------------------------------------------------------
def integrate_encounter(bvec, Vhat, V, s1, s2, m1, m2, Lfac=60.0, pinned=False,
                        rtol=1e-11, atol=1e-13):
    """Frozen-binary 3-body encounter (internal binary force OFF).

    pinned=False: bodies recoil under the perturber (the correct impulsive model).
    pinned=True : bodies held fixed; the would-be impulses are accumulated as
                  quadrature variables J1, J2 (this is the WRONG model, kept here to
                  demonstrate that it changes the close-encounter cutoff).

    Returns dict with final kicks dv1, dv2 and, at the closest approach to body 1,
    the osculating elements (vector beta1, V1) of the m*-body1 relative orbit."""
    r = np.linalg.norm(s2 - s1); L = Lfac*r
    xstar0 = bvec - L*Vhat
    hmax = np.inf        # DOP853 with rtol=1e-11 resolves the close pass adaptively
    tend = 2.2*L/V
    if pinned:
        # state: x*(3), v*(3), J1(3), J2(3)
        y0 = np.concatenate([xstar0, V*Vhat, np.zeros(3), np.zeros(3)])
        def rhs(t, y):
            xs = y[0:3]
            d1 = xs - s1; d2 = xs - s2
            r13 = (d1@d1)**1.5; r23 = (d2@d2)**1.5
            a_star = -G*m1*d1/r13 - G*m2*d2/r23
            j1dot = G*mstar*d1/r13; j2dot = G*mstar*d2/r23
            return np.concatenate([y[3:6], a_star, j1dot, j2dot])
        def ca(t, y):
            d = y[0:3]-s1; u = y[3:6]
            return d@u
        ca.direction = 1.0
        sol = solve_ivp(rhs, [0.0, tend], y0, method='DOP853', rtol=rtol, atol=atol,
                        max_step=hmax, events=ca)
        dv1 = sol.y[6:9,-1]; dv2 = sol.y[9:12,-1]
        ye = sol.y_events[0]
        seps = [np.linalg.norm(y[0:3]-s1) for y in ye]
        k = int(np.argmin(seps)); y = ye[k]
        d = y[0:3]-s1; u = y[3:6]
        mu1 = G*m1                      # pinned: source does not participate
    else:
        # state: x*(3), x1(3), x2(3), v*(3), v1(3), v2(3)
        y0 = np.concatenate([xstar0, s1, s2, V*Vhat, np.zeros(3), np.zeros(3)])
        def rhs(t, y):
            xs = y[0:3]; x1 = y[3:6]; x2 = y[6:9]
            d1 = xs - x1; d2 = xs - x2
            r13 = (d1@d1)**1.5; r23 = (d2@d2)**1.5
            a_star = -G*m1*d1/r13 - G*m2*d2/r23
            a_1 = G*mstar*d1/r13; a_2 = G*mstar*d2/r23
            return np.concatenate([y[9:12], y[12:15], y[15:18], a_star, a_1, a_2])
        def ca(t, y):
            d = y[0:3]-y[3:6]; u = y[9:12]-y[12:15]
            return d@u
        ca.direction = 1.0
        sol = solve_ivp(rhs, [0.0, tend], y0, method='DOP853', rtol=rtol, atol=atol,
                        max_step=hmax, events=ca)
        dv1 = sol.y[12:15,-1]; dv2 = sol.y[15:18,-1]
        ye = sol.y_events[0]
        seps = [np.linalg.norm(y[0:3]-y[3:6]) for y in ye]
        k = int(np.argmin(seps)); y = ye[k]
        d = y[0:3]-y[3:6]; u = y[9:12]-y[12:15]
        mu1 = G*(m1+mstar)              # recoil: reduced-mass Kepler parameter
    # osculating elements of the (relative) orbit at closest approach
    E = 0.5*(u@u) - mu1/np.linalg.norm(d)
    V1 = np.sqrt(2.0*E)
    Lvec = np.cross(d, u)
    Avec = np.cross(u, Lvec) - mu1*d/np.linalg.norm(d)   # Laplace-Runge-Lenz
    e = np.linalg.norm(Avec)/mu1
    phat = Avec/np.linalg.norm(Avec)
    Lhat = Lvec/np.linalg.norm(Lvec)
    qhat = np.cross(Lhat, phat)
    u_in = (phat + np.sqrt(e**2-1.0)*qhat)/e             # incoming asymptote direction
    beta1_vec = np.cross(u_in, Lvec)/V1                  # vector impact parameter
    return dict(dv1=dv1, dv2=dv2, beta1_vec=beta1_vec, V1=V1, u_in=u_in, mu1=mu1)

# --------------------------------------------------------------------------------------
print("="*88)
print("SECTION 1 -- the model matters: pinned bodies regularize at G m1/V^2,")
print("             recoiling bodies at G(m1+m*)/V^2   [Remark 'pinned vs recoil']")
print("="*88)
m1 = 1.0; V = 3.0
b90_recoil = G*(m1+mstar)/V**2; b90_pinned = G*m1/V**2
s1 = np.array([0.,0.,0.]); s2 = np.array([0.,0.,1.0])   # m2=0: body 2 inert
Vhat = np.array([1.,0.,0.])
print(f"  m1={m1}, m*={mstar}, V={V}:  G(m1+m*)/V^2={b90_recoil:.4f}   Gm1/V^2={b90_pinned:.4f}")
print(f"  {'beta/b90':>9} | {'c_recoil':>9} {'target':>8} | {'c_pinned':>9} {'target':>8}")
worst_r = worst_p = 0.0
Lfac1 = 3000.0
for f in [0.5, 1.0, 2.0, 4.0]:
    beta = f*b90_recoil
    bvec = np.array([0., beta, 0.])
    out = {}
    for tag, pin in [('recoil', False), ('pinned', True)]:
        d = integrate_encounter(bvec, Vhat, V, s1, s2, m1, 0.0, Lfac=Lfac1, pinned=pin)
        # correct for the finite launch distance L: the true asymptotic speed and
        # impact parameter differ from the launch values (energy/ang.mom. bookkeeping)
        mu = (G*(m1+mstar)) if not pin else (G*m1)
        Vinf = np.sqrt(V**2 - 2*mu/Lfac1)
        beta_inf = beta*V/Vinf
        dv2sq = d['dv1']@d['dv1']
        c2 = 4*G**2*mstar**2/(Vinf**2*dv2sq) - beta_inf**2
        out[tag] = np.sqrt(max(c2, 0.0))
    worst_r = max(worst_r, abs(out['recoil']-b90_recoil)/b90_recoil)
    worst_p = max(worst_p, abs(out['pinned']-b90_pinned)/b90_pinned)
    print(f"  {f:9.2f} | {out['recoil']:9.4f} {b90_recoil:8.4f} | {out['pinned']:9.4f} {b90_pinned:8.4f}")
ok1 = worst_r < 2e-2 and worst_p < 2e-2
print(f"  worst rel. err: recoil {worst_r:.1e}, pinned {worst_p:.1e}   {PASS(ok1)}")
print("  => the '+m*' in the cutoff comes from the recoil of the struck body;")
print("     'frozen binary' must mean internal force OFF, bodies dynamical.")
sys.stdout.flush()

# --------------------------------------------------------------------------------------
print("\n"+"="*88)
print("SECTION 2 -- deflection map b -> beta1* (VECTOR):  |delta| = O(b90) r-independent,")
print("             det(d beta1*/d b) = 1 + O(eta)       [Lemma 'Incoming-map regularity']")
print("="*88)
m1 = m2 = 1.0; V = 6.0
b90 = G*(m1+mstar)/V**2

def map_at(bvec, Vhat, V, s1, s2, m1, m2, eA, eB, Lfac=60.0):
    d = integrate_encounter(bvec, Vhat, V, s1, s2, m1, m2, Lfac=Lfac)
    return np.array([d['beta1_vec']@eA, d['beta1_vec']@eB]), d

t0 = time.time()
rows = []
for eta in [0.1, 0.05, 0.025]:
    r = b90/eta
    s1 = np.array([0.,0.,-0.5*r]); s2 = np.array([0.,0.,0.5*r])
    Vhat = np.array([1.,0.,0.])                    # perpendicular geometry (sin theta = 1)
    eA = np.array([0.,1.,0.]); eB = np.array([0.,0.,1.])   # impact-plane basis
    s1p = np.array([s1@eA, s1@eB])
    for f, tag in [(3.0, 'radial'), (3.0, 'tangential')]:
        if tag == 'radial':
            bvec0 = s1 + f*b90*np.array([0.,0.,1.])        # offset toward body 2
        else:
            bvec0 = s1 + f*b90*np.array([0.,1.,0.])        # offset out of the binary plane
        bvec0 = bvec0 - (bvec0@Vhat)*Vhat
        b2d0 = np.array([bvec0@eA, bvec0@eB])
        bm, dd = map_at(bvec0, Vhat, V, s1, s2, m1, m2, eA, eB)
        delta = bm - (b2d0 - s1p)
        # central-difference 2x2 Jacobian of the VECTOR map
        h = 0.05*b90; Jm = np.zeros((2,2))
        for i, e in enumerate([eA, eB]):
            bp, _ = map_at(bvec0 + h*e, Vhat, V, s1, s2, m1, m2, eA, eB)
            bmn, _ = map_at(bvec0 - h*e, Vhat, V, s1, s2, m1, m2, eA, eB)
            Jm[:,i] = (bp - bmn)/(2*h)
        detJ = np.linalg.det(Jm)
        rows.append((eta, tag, np.linalg.norm(delta)/b90, detJ-1.0, dd['V1']/V-1.0, dd))
        print(f"  eta={eta:6.3f} {tag:10s} beta1/b90={f:.1f}:  |delta|/b90={rows[-1][2]:6.3f}"
              f"   detJ-1={rows[-1][3]:+9.5f}   (detJ-1)/eta={rows[-1][3]/eta:+7.3f}"
              f"   V1/V-1={rows[-1][4]:+8.5f}")
    sys.stdout.flush()
# scaling checks
print("\n  scaling as eta halves (expect |delta|/b90 -> const, (detJ-1) ratio -> 2):")
ok_delta = ok_jac = True
for tag in ['radial', 'tangential']:
    sel = [row for row in rows if row[1] == tag]
    ds = [row[2] for row in sel]; js = [row[3] for row in sel]
    dvar = (max(ds)-min(ds))/np.mean(ds)
    ratios = [js[i]/js[i+1] for i in range(len(js)-1) if abs(js[i+1]) > 1e-12]
    print(f"   {tag:10s}: |delta|/b90 = {', '.join(f'{d:.3f}' for d in ds)}  (spread {dvar:.0%});"
          f"  (detJ-1) ratios = {', '.join(f'{q:.2f}' for q in ratios)}")
    ok_delta &= dvar < 0.35
    ok_jac &= all(1.5 < q < 2.7 for q in ratios)
print(f"  |delta| ~ b90, r-independent: {PASS(ok_delta)};   detJ-1 = O(eta): {PASS(ok_jac)}")
print(f"  (elapsed {time.time()-t0:.0f}s)")
sys.stdout.flush()

# --------------------------------------------------------------------------------------
print("\n"+"="*88)
print("SECTION 3 -- kick fidelity: |dv1|^2 (beta1*^2 + b90(V1*)^2) V1*^2 / (4 G^2 m*^2) = 1+eps")
print("             and the naive asymptotic-V version fits worse   [Lemma 'Kick fidelity']")
print("="*88)
print(f"  {'eta':>7} {'geometry':>10} | {'eps (osculating)':>17} | {'eps (asympt. V)':>16}")
ok_fid = True; pairs = []
for eta, tag, _, _, _, dd in rows:
    beta1 = np.linalg.norm(dd['beta1_vec']); V1 = dd['V1']
    b90_loc = G*(m1+mstar)/V1**2
    dv2sq = dd['dv1']@dd['dv1']
    eps_osc = dv2sq*(beta1**2 + b90_loc**2)*V1**2/(4*G**2*mstar**2) - 1.0
    eps_nai = dv2sq*(beta1**2 + b90**2)*V**2/(4*G**2*mstar**2) - 1.0
    pairs.append((eta, tag, eps_osc, eps_nai))
    print(f"  {eta:7.3f} {tag:>10} | {eps_osc:+17.5f} | {eps_nai:+16.5f}")
for tag in ['radial', 'tangential']:
    ee = [abs(p[2]) for p in pairs if p[1] == tag]
    ok_fid &= ee[0] > ee[-1] and ee[-1] < 0.05
    nn = [abs(p[3]) for p in pairs if p[1] == tag]
    ok_fid &= all(n > e for n, e in zip(nn, ee))
print(f"  osculating eps shrinks with eta and beats the asymptotic-V fit: {PASS(ok_fid)}")
print("  => the exact kick IS the two-body kick at the osculating (beta1*, V1*), incl. the")
print("     (V/V1*)^2 prefactor and the b90(V1*) shift; corrections vanish in the limit.")

print("\nDone.")
