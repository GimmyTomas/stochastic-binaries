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

  4. THE DIRECT IMPULSE CANCELS (Lemma "Kick fidelity", exact identity): the companion's
     direct impulse on the perturber, A = (m*/(m1+m*)) \int g2 dt, does NOT appear in the
     body-1 kick:  dv1 - dv1_2B is the tidal path distortion alone, of relative size
     O(eta^{3/2}), while |A| = O(eta).  We measure |dv1 - dv1_2B| << |A| (ratio ~ sqrt(eta)),
     and show that SUBTRACTING A (as if it were present) leaves a residual of the full |A|
     size -- i.e. a decomposition dv1 = dv1_2B(1+eps) + A would be wrong.

  5. LENSING CAUSTIC (Lemma "All orientations: exit-map measure estimate"): with the binary
     aligned along V, the upstream body is a gravitational lens with Einstein radius
     bE = sqrt(2 G m2 r / V^2) = O(sqrt(b90 r)).  An annulus of asymptotic impact parameters
     at bE focuses onto the downstream body: we scan beta and find a direct hit
     (pericenter ~ 0) at beta ~ bE >> b90, with |d peri/d beta| = O(1) across the ring.
     Hence mu{pericenter <= s} ~ 2 pi bE s there, the caustic term of the measure estimate.

  6. RAINBOW SCATTERING of a heavy perturber (Lemma "Sequential encounters" and the
     lab-angle formula): for m* > m1, the perturber's LAB deflection angle off a single
     recoiling body,  tan(chi) = m1 sin(Theta) / (m* + m1 cos(Theta))  with Theta the
     relative-orbit Rutherford angle, attains a maximum chi_max = arcsin(m1/m*) at
     beta_rb = b90 sqrt((m*-m1)/(m*+m1)) -- a fold (rainbow) where the lab cross-section
     diverges.  Consequently the downstream miss map g(beta) (transverse offset at a lab
     plane a distance ell behind the scatterer) is NON-monotone below b90 while g' >= 1
     above it, and its Einstein zero sits at sqrt(2 G m1 ell)/V (bare lens mass), not at
     sqrt(2 b90 ell).  We measure chi(beta) and g(beta) from direct integrations.

G = 1 throughout; m* = 1 in Sections 1-5 and m* = 3 in Section 6 (heavy perturber).
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
sys.stdout.flush()

# --------------------------------------------------------------------------------------
def encounter_extra(bvec, Vhat, V, s1, s2, m1, m2, Lfac, rtol=1e-11, atol=1e-13,
                    track_min1=False):
    """Frozen-binary 3-body encounter with two extras needed by Sections 4-5:
    the quadrature J2 = int g2 dt (companion's direct pull on the relative coordinate)
    and, optionally, the minimum distance to body 1 along the whole trajectory.
    Also returns the ANALYTIC isolated-2B kick vector at the osculating pericenter
    elements: dv1_2B = (2 G m* V1 / (mu1 e)) p_hat  (p_hat = unit Laplace-Runge-Lenz),
    which avoids any truncation error in the reference kick."""
    r = np.linalg.norm(s2 - s1); L = Lfac * r
    x0 = bvec - L * Vhat
    y0 = np.concatenate([x0, s1, s2, V*Vhat, np.zeros(3), np.zeros(3), np.zeros(3)])
    tend = 2.4 * L / V
    def rhs(t, y):
        xs = y[0:3]; x1 = y[3:6]; x2 = y[6:9]
        d1 = xs - x1; d2 = xs - x2
        r13 = (d1 @ d1) ** 1.5; r23 = (d2 @ d2) ** 1.5
        a_star = -G*m1*d1/r13 - G*m2*d2/r23
        a1 = G*mstar*d1/r13; a2 = G*mstar*d2/r23
        g2 = -G*m2*d2/r23
        return np.concatenate([y[9:12], y[12:15], y[15:18], a_star, a1, a2, g2])
    def ca1(t, y):
        d = y[0:3] - y[3:6]; u = y[9:12] - y[12:15]
        return d @ u
    ca1.direction = 1.0
    sol = solve_ivp(rhs, [0.0, tend], y0, method='DOP853', rtol=rtol, atol=atol,
                    events=ca1, dense_output=track_min1)
    res = dict(dv1=sol.y[12:15, -1], dv2=sol.y[15:18, -1], J2=sol.y[18:21, -1])
    ye = sol.y_events[0]
    if len(ye) > 0:
        seps = [np.linalg.norm(y[0:3] - y[3:6]) for y in ye]
        k = int(np.argmin(seps)); y = ye[k]
        d = y[0:3] - y[3:6]; u = y[9:12] - y[12:15]
        mu1 = G * (m1 + mstar)
        E = 0.5*(u @ u) - mu1/np.linalg.norm(d)
        V1 = np.sqrt(2.0*E)
        Lv = np.cross(d, u)
        Av = np.cross(u, Lv) - mu1*d/np.linalg.norm(d)
        e = np.linalg.norm(Av)/mu1
        phat = Av/np.linalg.norm(Av)
        res.update(V1=V1, e=e, peri=min(seps),
                   dv1_2B=(2.0*G*mstar*V1/(mu1*e))*phat)
    if track_min1:
        ts = np.linspace(0.0, sol.t[-1], 4000)
        Z = sol.sol(ts)
        dmin = np.min(np.linalg.norm(Z[0:3] - Z[3:6], axis=0))
        res['mind1'] = min(dmin, res.get('peri', np.inf))
    return res

print("\n"+"="*88)
print("SECTION 4 -- the companion's direct impulse CANCELS in the body-1 kick")
print("             [Lemma 'Kick fidelity', exact identity]")
print("  D1 := dv1 - dv1_2B (measured)   A := (m*/(m1+m*)) int g2 dt (the direct impulse)")
print("  exact identity => |D1| << |A| (D1 is the tidal path distortion, O(eta^{3/2}) vs")
print("  A = O(eta)); a decomposition carrying A additively would instead give |D1-A| << |A|.")
print("="*88)
m1 = m2 = 1.0; V = 6.0
b90 = G*(m1+mstar)/V**2
bnt = G*(m1+m2+mstar)/V**2
Vhat = np.array([1., 0., 0.])
print(f"  {'eta':>7} {'offset':>11} | {'|A|':>9} {'|D1|':>10} | {'|D1|/|A|':>9} {'|D1-A|/|A|':>11}")
t0 = time.time()
ratios_DA = {'radial': [], 'tangential': []}
ok4 = True
for eta in [0.04, 0.02, 0.01, 0.005]:
    r = bnt/eta
    s1 = np.array([0., 0., -0.5*r]); s2 = np.array([0., 0., 0.5*r])
    for tag, off in [('radial', np.array([0., 0., 1.])),
                     ('tangential', np.array([0., 1., 0.]))]:
        bvec = s1 + 3.0*b90*off
        bvec = bvec - (bvec @ Vhat)*Vhat
        d = encounter_extra(bvec, Vhat, V, s1, s2, m1, m2, Lfac=400.0)
        A = (mstar/(m1+mstar))*d['J2']
        D1 = d['dv1'] - d['dv1_2B']
        nA = np.linalg.norm(A); nD = np.linalg.norm(D1)
        nDA = np.linalg.norm(D1 - A)
        ratios_DA[tag].append(nD/nA)
        ok4 &= nDA/nA > 0.8                      # subtracting A leaves the full |A|
        print(f"  {eta:7.3f} {tag:>11} | {nA:9.6f} {nD:10.6f} | {nD/nA:9.4f} {nDA/nA:11.4f}")
    sys.stdout.flush()
for tag in ['radial', 'tangential']:
    rr = ratios_DA[tag]
    ok4 &= all(rr[i] > rr[i+1] for i in range(len(rr)-1)) and rr[-1] < 0.02
print(f"  |D1|/|A| falls monotonically (~sqrt(eta)) and |D1-A| stays ~|A|: {PASS(ok4)}")
print("  => the direct impulse is absent from the kick: the companion acts on dv1 only")
print("     through the relabeling (beta1*, V1*) and the O(eta^{3/2}) tidal distortion.")
print(f"  (elapsed {time.time()-t0:.0f}s)")
sys.stdout.flush()

# --------------------------------------------------------------------------------------
print("\n"+"="*88)
print("SECTION 5 -- lensing caustic at the Einstein radius (aligned binary)")
print("             [Lemma 'All orientations: exit-map measure estimate']")
print("  Bodies aligned with V, body 2 upstream, body 1 a distance r downstream.")
print("  Scan asymptotic beta: pericenter w.r.t. body 1 must dip to ~0 at")
print("  beta ~ bE = sqrt(2 G m2 r/V^2) >> b90, with O(1) slope across the ring.")
print("="*88)
m1 = m2 = 1.0; V = 6.0
b90 = G*(m1+mstar)/V**2
r = 200.0*b90
bE = np.sqrt(2.0*G*m2*r/V**2)
s2 = np.array([0., 0., 0.]); s1 = np.array([0., 0., r])      # body 2 met FIRST
Vhat = np.array([0., 0., 1.])
print(f"  r = 200 b90:  bE = {bE/b90:.1f} b90")
t0 = time.time()
grid = np.linspace(0.8*bE, 1.3*bE, 11)
peris = []
for beta in grid:
    d = encounter_extra(np.array([beta, 0., 0.]), Vhat, V, s1, s2, m1, m2,
                        Lfac=40.0, rtol=1e-10, track_min1=True)
    peris.append(d['mind1'])
    print(f"    beta = {beta/b90:6.2f} b90   min dist to body 1 = {d['mind1']/b90:8.3f} b90")
    sys.stdout.flush()
k = int(np.argmin(peris))
lo, hi = grid[max(k-1, 0)], grid[min(k+1, len(grid)-1)]
for _ in range(14):
    mids = np.linspace(lo, hi, 5)
    vals = [encounter_extra(np.array([b, 0., 0.]), Vhat, V, s1, s2, m1, m2,
                            Lfac=40.0, rtol=1e-10, track_min1=True)['mind1'] for b in mids]
    j = int(np.argmin(vals))
    lo, hi = mids[max(j-1, 0)], mids[min(j+1, len(mids)-1)]
    if (hi - lo) < 1e-3*b90:
        break
beta_star = 0.5*(lo + hi)
p_star = encounter_extra(np.array([beta_star, 0., 0.]), Vhat, V, s1, s2, m1, m2,
                         Lfac=40.0, rtol=1e-10, track_min1=True)['mind1']
h = 0.5*b90
p_off = encounter_extra(np.array([beta_star + h, 0., 0.]), Vhat, V, s1, s2, m1, m2,
                        Lfac=40.0, rtol=1e-10, track_min1=True)['mind1']
slope = abs(p_off - p_star)/h
ok5 = (p_star < 0.05*b90) and (0.75 < beta_star/bE < 1.25) and (0.2 < slope < 5.0)
print(f"\n  caustic: beta* = {beta_star/b90:.2f} b90 = {beta_star/bE:.3f} bE,"
      f"  pericenter(body 1) = {p_star/b90:.4f} b90,  |d peri/d beta| ~ {slope:.2f}")
print(f"  direct hit at beta ~ bE >> b90 with O(1) slope: {PASS(ok5)}")
print("  => mu{pericenter <= s} ~ 2 pi bE s / slope on the ring: the caustic term of the")
print("     measure estimate; no O(s^2) bound can hold near alignment.  Only the O(eta)")
print("     solid angle of aligned orientations keeps this out of the theta-averages.")
print(f"  (elapsed {time.time()-t0:.0f}s)")
sys.stdout.flush()

# --------------------------------------------------------------------------------------
print("\n"+"="*88)
print("SECTION 6 -- rainbow scattering of a heavy perturber (m* > m1)")
print("             [Lemma 'Sequential encounters'; lab-angle formula]")
print("  Measured lab deflection chi(beta) off a single recoiling body must follow")
print("  tan chi = m1 sin(Theta)/(m* + m1 cos(Theta)), peak at chi_max = arcsin(m1/m*),")
print("  beta_rb = b90 sqrt((m*-m1)/(m*+m1)); the downstream miss map is non-monotone")
print("  below b90, has g' >= 1 above, and its Einstein zero is sqrt(2 G m1 ell)/V.")
print("="*88)
mstar = 3.0                       # heavy perturber for this section (module-level global)
m1_rb = 1.0; V = 6.0
b90_rb = G*(m1_rb+mstar)/V**2
zhat = np.array([0., 0., 1.])

def lab_two_body(beta, ell=None, Lstart=None, rtol=1e-10, atol=1e-12):
    """Perturber (mass mstar) incoming along +z at impact parameter beta; single free
    body of mass m1_rb at rest at the origin.  Returns the measured asymptotic lab
    deflection angle chi of the perturber and, if ell is given, the transverse offset
    ('miss') where the perturber crosses the lab plane z = ell."""
    Lstart = Lstart if Lstart is not None else (2000.0*b90_rb if ell is None else 1000.0*b90_rb)
    y0 = np.concatenate([np.array([beta, 0., -Lstart]), np.zeros(3),
                         V*zhat, np.zeros(3)])                    # x*, x1, v*, v1
    def rhs(t, y):
        d = y[0:3] - y[3:6]
        r3 = (d@d)**1.5
        return np.concatenate([y[6:9], y[9:12], -G*m1_rb*d/r3, G*mstar*d/r3])
    events = []
    if ell is not None:
        def cross(t, y): return y[2] - ell
        cross.terminal = True; cross.direction = 1.0
        events.append(cross)
    tend = 6.0*(Lstart + (ell if ell is not None else Lstart))/V
    sol = solve_ivp(rhs, [0.0, tend], y0, method='DOP853', rtol=rtol, atol=atol,
                    events=events or None)
    miss = None
    if ell is not None and sol.t_events[0].size:
        miss = sol.y_events[0][0][0]              # x-coordinate at the plane (planar orbit)
    vf = sol.y[6:9, -1]
    chi = float(np.arccos(np.clip(vf@zhat/np.linalg.norm(vf), -1.0, 1.0)))
    # launch-distance corrections for the analytic comparison
    D0 = np.hypot(beta, Lstart)
    Vinf = np.sqrt(V**2 - 2*G*(m1_rb+mstar)/D0)
    beta_inf = beta*V/Vinf
    return chi, miss, Vinf, beta_inf

def chi_formula(beta_inf, Vinf):
    Theta = 2.0*np.arctan(G*(m1_rb+mstar)/Vinf**2/beta_inf)
    return float(np.arctan2(m1_rb*np.sin(Theta), mstar + m1_rb*np.cos(Theta)))

chi_max_pred = np.arcsin(m1_rb/mstar)
beta_rb_pred = b90_rb*np.sqrt((mstar-m1_rb)/(mstar+m1_rb))
print(f"  m1={m1_rb}, m*={mstar}, V={V}:  b90={b90_rb:.4f},"
      f"  chi_max={chi_max_pred:.4f} rad at beta_rb={beta_rb_pred/b90_rb:.3f} b90")
t0 = time.time()
grid6 = np.geomspace(0.2, 8.0, 33)*b90_rb
chis, preds = [], []
for beta in grid6:
    chi, _, Vinf, beta_inf = lab_two_body(beta)
    chis.append(chi); preds.append(chi_formula(beta_inf, Vinf))
chis = np.array(chis); preds = np.array(preds)
err_chi = np.max(np.abs(chis - preds))
k6 = int(np.argmax(chis))
chi_max_meas = chis[k6]; beta_rb_meas = grid6[k6]
print(f"  max |chi_measured - chi_formula| over {len(grid6)} betas: {err_chi:.2e} rad")
print(f"  measured peak: chi_max={chi_max_meas:.4f} at beta={beta_rb_meas/b90_rb:.3f} b90"
      f"   (predicted {chi_max_pred:.4f} at {beta_rb_pred/b90_rb:.3f} b90)")
ok6a = err_chi < 3e-3
ok6b = abs(chi_max_meas - chi_max_pred) < 5e-3 and \
       abs(beta_rb_meas/beta_rb_pred - 1.0) < 0.15
print(f"  lab-angle formula reproduced: {PASS(ok6a)};  rainbow fold located: {PASS(ok6b)}")
sys.stdout.flush()

ell = 200.0*b90_rb
bE_rb = np.sqrt(2.0*G*m1_rb*ell)/V                 # bare lens mass m1
bE_wrong = np.sqrt(2.0*b90_rb*ell)                 # the (wrong) reduced-mass version
print(f"\n  miss map at ell = 200 b90:  Einstein zero predicted at sqrt(2 G m1 ell)/V"
      f" = {bE_rb/b90_rb:.2f} b90   [sqrt(2 b90 ell) would be {bE_wrong/b90_rb:.2f} b90]")
grid_m = np.array([0.30, 0.45, 0.60, 0.75, 0.90, 1.10, 1.40, 2.0, 3.0, 5.0, 8.0,
                   9.0, 10.0, 11.0, 12.0])*b90_rb
misses = []
for beta in grid_m:
    _, miss, _, _ = lab_two_body(beta, ell=ell)
    misses.append(miss)
misses = np.array(misses)
slopes = np.diff(misses)/np.diff(grid_m)
low = grid_m[:-1] < 0.95*b90_rb
high = grid_m[:-1] >= 1.05*b90_rb
ok6c = np.min(slopes[low]) < 0.0                    # non-monotone (plunge) below b90
ok6d = np.min(slopes[high]) > 0.9                   # g' >= 1 above b90 (finite-ell tolerance)
for b, m in zip(grid_m, misses):
    print(f"    beta = {b/b90_rb:6.2f} b90   miss(z=ell) = {m/b90_rb:+9.2f} b90")
# bisect the Einstein zero on the monotone branch
lo6, hi6 = 8.0*b90_rb, 12.0*b90_rb
for _ in range(24):
    mid = 0.5*(lo6 + hi6)
    _, mm, _, _ = lab_two_body(mid, ell=ell)
    if mm < 0.0: lo6 = mid
    else: hi6 = mid
    if hi6 - lo6 < 1e-3*b90_rb:
        break
beta0 = 0.5*(lo6 + hi6)
ok6e = abs(beta0/bE_rb - 1.0) < 0.10 and beta0/bE_wrong < 0.65
print(f"\n  plunge branch below b90 (min slope {np.min(slopes[low]):+.1f} < 0): {PASS(ok6c)};"
      f"   g' >= 1 above b90 (min slope {np.min(slopes[high]):+.2f}): {PASS(ok6d)}")
print(f"  Einstein zero measured at beta0 = {beta0/b90_rb:.2f} b90 = {beta0/bE_rb:.3f} x"
      f" sqrt(2 G m1 ell)/V   (= {beta0/bE_wrong:.3f} x the wrong sqrt(2 b90 ell)): {PASS(ok6e)}")
print("  => the targeting map is the LAB deflection: monotone with g' >= 1 only above b90,")
print("     rainbow fold below (the caustic terms of the sequential-encounter lemma), and")
print("     the Einstein ring carries the bare lens mass -- recoil bookkeeping again.")
print(f"  (elapsed {time.time()-t0:.0f}s)")

print("\nDone.")
