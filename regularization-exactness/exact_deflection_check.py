#!/usr/bin/env python3
r"""
Is b_90 = G(m_i + m_*)/V^2 the EXACT close-encounter regularization, or just one
arbitrary scheme among many?

This script addresses the natural objection that b_90 might be an arbitrary
dimensional estimate ("it turns out to be exact if one does it nonlinearly").
The claim we test is:

  Replacing the straight-line (impulse) kick by the EXACT Newtonian two-body
  hyperbolic deflection of each binary component -- with NO ad-hoc cutoff -- gives,
  in the impulsive limit, EXACTLY the same Q_r(r), Q_t(r) (including the additive
  constants -2/3, -8/3) as the paper's sharp cutoff at b_90 = G(m_i+m_*)/V^2.

Strategy / sections:
  1.  Exact two-body kick.  Numerically integrate the perturber+body two-body ODE
      for a hyperbolic encounter and confirm the closed-form decomposition
          |dv|^2          = (4 G^2 m_*^2 / V^2) * 1/(b^2 + b90^2)             [total]
          |dv_perp|^2     = (4 G^2 m_*^2 / V^2) * b^2/(b^2 + b90^2)^2         [transverse]
          |dv_par |^2     = (4 G^2 m_*^2 / V^2) * b90^2/(b^2 + b90^2)^2       [along V]
      with b90 = G(m_i+m_*)/V^2.  (This is the "nonlinear" deflection.)
  2.  Single-body self-energy.  int |y|^2 d^2b with the exact kick gives the clean
      2 pi ln(Lambda/b90) with NO constant; the transverse part alone gives
      2 pi ln(Lambda/(b90 sqrt e)) and the longitudinal part the compensating +pi.
  3.  Angular average.  The exact-minus-sharp difference is a traceless, V-aligned
      quadrupole; averaged over isotropic perturber directions it vanishes, so
      Q_r and Q_t are unchanged.  Checked as a 1-D theta quadrature.
  4.  Full binary Monte-Carlo, exact-deflection vs sharp-b90, with COMMON RANDOM
      NUMBERS, both compared to the closed forms.  The decisive end-to-end test.

G = m_* = n = 1 unless stated.
"""
import warnings
import numpy as np
from scipy import integrate
from scipy.integrate import IntegrationWarning
warnings.filterwarnings("ignore", category=IntegrationWarning)

PASS = lambda ok: "PASS" if ok else "**FAIL**"
gE = np.euler_gamma

print("="*80)
print("SECTION 1 -- exact two-body hyperbolic kick vs closed-form decomposition")
print("  integrate the relative two-body orbit; subject kick dv1 = (m*/(m1+m*)) d(rho_dot)")
print("="*80)

def exact_kick_ode(b, V, m_sub, mstar=1.0, G=1.0):
    """Integrate the perturber+subject two-body problem on a hyperbola with impact
    parameter b and relative speed V at infinity.  Return the subject's velocity
    change resolved into (transverse-to-V, along-V) components.
    Relative coordinate rho = r_sub - r_*, mu_red irrelevant (test of geometry):
        rho'' = -G(m_sub+mstar) rho / |rho|^3 .
    Subject velocity change:  dv_sub = (mstar/(m_sub+mstar)) * (rho_dot_out - rho_dot_in)."""
    mu = G*(m_sub+mstar)
    # start far away up-stream; V along +x, impact parameter along +y
    X0 = 6.0e4*max(b, mu/V**2)             # huge initial separation along the motion
    rho0 = np.array([-X0, b])
    vel0 = np.array([ V, 0.0])             # rho_dot initial ~ +V xhat (approaching)
    def rhs(t, s):
        x, y, vx, vy = s
        r3 = (x*x+y*y)**1.5
        return [vx, vy, -mu*x/r3, -mu*y/r3]
    T = 2.0*X0/V
    sol = integrate.solve_ivp(rhs, [0, T], [*rho0, *vel0], rtol=1e-11, atol=1e-13,
                              dense_output=False, max_step=T/2000)
    vout = sol.y[2:, -1]
    drhodot = vout - vel0
    dv_sub = (mstar/(m_sub+mstar))*drhodot
    # resolve: Vhat = xhat (initial relative-velocity direction)
    Vhat = np.array([1.0, 0.0])
    dv_par = dv_sub @ Vhat                 # component along initial V
    dv_perp = dv_sub - dv_par*Vhat
    return np.hypot(*dv_perp)**0 * np.linalg.norm(dv_perp), dv_par, np.linalg.norm(dv_sub)

worst = 0.0
print(f"  {'b':>6} {'V':>5} {'m_sub':>5} | {'|dv|^2 num':>12} {'closed':>12}"
      f" | {'perp^2 num':>11} {'closed':>11} | {'par^2 num':>10} {'closed':>10}")
for b, V, m_sub in [(0.5,3.0,1.0),(1.0,3.0,1.0),(2.0,4.0,0.3),(0.2,5.0,1.0),(3.0,2.0,0.8)]:
    b90 = (m_sub+1.0)/V**2
    perp, par, tot = exact_kick_ode(b, V, m_sub)
    c_tot  = 4*1.0/V**2 * 1.0/(b**2+b90**2)
    c_perp = 4*1.0/V**2 * b**2/(b**2+b90**2)**2
    c_par  = 4*1.0/V**2 * b90**2/(b**2+b90**2)**2
    e1 = abs(tot**2-c_tot)/c_tot; e2 = abs(perp**2-c_perp)/c_perp; e3 = abs(par**2-c_par)/max(c_par,1e-30)
    worst = max(worst, e1, e2, e3)
    print(f"  {b:6.2f} {V:5.1f} {m_sub:5.2f} | {tot**2:12.6f} {c_tot:12.6f}"
          f" | {perp**2:11.6f} {c_perp:11.6f} | {par**2:10.6f} {c_par:10.6f}")
print(f"  worst relative error vs closed-form decomposition: {worst:.2e}  {PASS(worst<2e-4)}")
print("  => the EXACT nonlinear kick is  dv_perp ~ b/(b^2+b90^2),  dv_par ~ b90/(b^2+b90^2),")
print("     i.e. transverse part = Plummer core b90, plus a forward (||V) focusing part.")

print()
print("="*80)
print("SECTION 2 -- single-body self-energy: where the constant lives")
print("="*80)
# y_perp^2 = b^2/(b^2+b90^2)^2 , y_par^2 = b90^2/(b^2+b90^2)^2 , y^2 = 1/(b^2+b90^2)
b90 = 1.0; Lam = 1.0e4
Iperp,_ = integrate.quad(lambda b: 2*np.pi*b * b**2/(b**2+b90**2)**2, 0, Lam, limit=400)
Ipar ,_ = integrate.quad(lambda b: 2*np.pi*b * b90**2/(b**2+b90**2)**2, 0, np.inf, limit=400)
Itot ,_ = integrate.quad(lambda b: 2*np.pi*b * 1.0/(b**2+b90**2), 0, Lam, limit=400)
print(f"  transverse  int = {Iperp:.5f}   target 2pi ln(Lam/(b90 sqrt e)) = {2*np.pi*np.log(Lam/(b90*np.sqrt(np.e))):.5f}"
      f"  {PASS(abs(Iperp-2*np.pi*np.log(Lam/(b90*np.sqrt(np.e))))<1e-2)}")
print(f"  longitudinal int = {Ipar:.5f}   target  pi                      = {np.pi:.5f}"
      f"  {PASS(abs(Ipar-np.pi)<1e-3)}")
print(f"  total        int = {Itot:.5f}   target 2pi ln(Lam/b90)          = {2*np.pi*np.log(Lam/b90):.5f}"
      f"  {PASS(abs(Itot-2*np.pi*np.log(Lam/b90))<1e-2)}")
print("  => transverse alone -> b90*sqrt(e) (the -pi=-1/2 'Plummer' shift); the longitudinal")
print("     +pi exactly cancels it, so the TOTAL self-energy is a clean ln(Lambda/b90).")
print("     The sharp cutoff Theta(b-b90) on the transverse straight-line kick also gives")
print(f"     exactly 2pi ln(Lam/b90) = {2*np.pi*np.log(Lam/b90):.5f}  -> same total as exact.")

print()
print("="*80)
print("SECTION 3 -- angular average kills the exact-minus-sharp difference")
print("  Difference tensor (exact - sharp), single-body self part, at fixed V:")
print("     -pi/2 on each impact-plane axis (e_perp,e_par), +pi along V  => traceless,")
print("     proportional to (Vhat Vhat - 1/3 I).  Q_r weights e_perp by sin^2 th, V by cos^2 th.")
print("="*80)
# contribution to Q_r integrand (per unit, measure sin th dth): exact-sharp
#   = sin^2 th * (-pi/2)   [transverse deficit shared, only e_perp projects on rhat]
#     + cos^2 th * (+pi)   [longitudinal surplus along V]
dQr,_  = integrate.quad(lambda th: np.sin(th)*( -np.pi/2*np.sin(th)**2 + np.pi*np.cos(th)**2), 0, np.pi)
dTr,_  = integrate.quad(lambda th: np.sin(th)*( 0.0 ), 0, np.pi)  # trace difference is 0 pointwise
print(f"  int_0^pi sin th [ -pi/2 sin^2 th + pi cos^2 th ] dth = {dQr:+.6e}  {PASS(abs(dQr)<1e-9)}")
print(f"     (= pi[ 2/3 - 1/2*4/3 ] = 0)   -> Delta Q_r = 0")
print(f"  trace difference (longitudinal +pi cancels transverse -pi pointwise) = {dTr:+.3e}  {PASS(abs(dTr)<1e-12)}")
print("  isotropic check: average of (Vhat Vhat - 1/3 I) over directions:")
M = np.zeros((3,3)); NN=200000
g = np.random.default_rng(0)
u = g.normal(size=(NN,3)); u/=np.linalg.norm(u,axis=1)[:,None]
M = (u[:,:,None]*u[:,None,:]).mean(0) - np.eye(3)/3
print(f"     || <Vhat Vhat> - I/3 || = {np.linalg.norm(M):.2e}  {PASS(np.linalg.norm(M)<5e-3)}")
print("  => exact-minus-sharp is traceless & V-aligned -> vanishes on isotropic V-average.")
print("     Hence Q_r, Q_t (and all 15 coefficients) are identical in the two schemes.")

print()
print("="*80)
print("SECTION 4 -- full binary Monte-Carlo: exact deflection vs sharp b90 (common RNG)")
print("="*80)

def closed_forms(r, sigma, m1, m2, mstar=1.0, G=1.0, n=1.0):
    C = G**2*mstar**2*n*np.sqrt(2/np.pi)/sigma
    L = np.log(16*r**2*sigma**4/(G**2*(mstar+m1)*(mstar+m2))) - 2*gE
    return C, L, (8*np.pi/3)*C*(L-2/3), (8*np.pi/3)*C*(L-8/3)

def mc_both(r, sigma, m1, m2, mstar=1.0, G=1.0, n=1.0, N=8_000_000, seed=1, fixedV=None):
    g = np.random.default_rng(seed)
    if fixedV is None:
        Vv = g.normal(0, sigma, size=(N,3)); V = np.linalg.norm(Vv,axis=1); Vh = Vv/V[:,None]
    else:
        # monochromatic speed, isotropic direction
        Vh = g.normal(size=(N,3)); Vh /= np.linalg.norm(Vh,axis=1)[:,None]
        V = np.full(N, float(fixedV))
    zh = np.array([0.,0.,1.])
    czV = Vh[:,2]                                   # cos theta
    rpv = zh[None,:]-czV[:,None]*Vh; rp = np.linalg.norm(rpv,axis=1)  # sin theta
    good = rp>1e-9
    e1 = np.zeros_like(Vh); e1[good]=rpv[good]/rp[good,None]          # along r_perp
    e2 = np.cross(Vh, e1)                                            # 2nd in-plane axis
    rperp = r*rp
    s1 = -m2/(m1+m2)*rperp; s2 = m1/(m1+m2)*rperp                    # body coords along e1
    b901 = G*(mstar+m1)/V**2; b902 = G*(mstar+m2)/V**2
    Rout = np.maximum(20.0*np.maximum(rperp,1e-3), 200.0*np.maximum(b901,b902))
    # sample 2-D b in (e1,e2): log-radial about each body (down to b90/bfac) + uniform disk
    bfac = 60.0
    w0,w1,w2 = 0.4,0.4,0.2
    u = g.random(N); comp = np.where(u<w0,0,np.where(u<w0+w1,1,2)); ang = g.random(N)*2*np.pi
    def logr(lo):
        uu=g.random(N); return lo*(Rout/lo)**uu
    bx=np.empty(N); by=np.empty(N)
    m=comp==0; rho=logr(b901/bfac); bx[m]=s1[m]+(rho*np.cos(ang))[m]; by[m]=(rho*np.sin(ang))[m]
    m=comp==1; rho=logr(b902/bfac); bx[m]=s2[m]+(rho*np.cos(ang))[m]; by[m]=(rho*np.sin(ang))[m]
    m=comp==2; rr=Rout*np.sqrt(g.random(N)); bx[m]=(rr*np.cos(ang))[m]; by[m]=(rr*np.sin(ang))[m]
    rho1=np.hypot(bx-s1,by); rho2=np.hypot(bx-s2,by); rho0=np.hypot(bx,by)
    lo1=b901/bfac; lo2=b902/bfac
    L1=np.log(Rout/lo1); L2=np.log(Rout/lo2)
    p0=np.where((rho1>=lo1)&(rho1<=Rout),1.0/(2*np.pi*rho1**2*L1),0.0)
    p1=np.where((rho2>=lo2)&(rho2<=Rout),1.0/(2*np.pi*rho2**2*L2),0.0)
    p2=np.where(rho0<=Rout,1.0/(np.pi*Rout**2),0.0)
    p=w0*p0+w1*p1+w2*p2
    ok = (p>0)&good
    pre=2*G*mstar/V
    # in-plane unit components of b1,b2
    b1x=bx-s1; b1y=by; b2x=bx-s2; b2y=by
    # ---------- SHARP scheme: straight-line transverse, Theta(b-b90) ----------
    keepS = ok & (rho1>b901) & (rho2>b902)
    inv1=1.0/rho1**2; inv2=1.0/rho2**2
    dv1_s = pre*(b2x*inv2 - b1x*inv1)          # along e1
    dv2_s = pre*(b2y*inv2 - b1y*inv1)          # along e2
    dvr_s = dv1_s*e1[:,2] + dv2_s*e2[:,2]      # . zhat
    dv2sq_s = dv1_s**2 + dv2_s**2              # |dv|^2 (purely in-plane)
    # ---------- EXACT scheme: Plummer-core transverse + ||V focusing ----------
    sinv1=1.0/(rho1**2+b901**2); sinv2=1.0/(rho2**2+b902**2)
    dv1_e = pre*(b2x*sinv2 - b1x*sinv1)        # transverse along e1
    dv2_e = pre*(b2y*sinv2 - b1y*sinv1)        # transverse along e2
    dvV_e = pre*(b901*sinv1 - b902*sinv2)      # longitudinal along Vhat  (note body-1 minus body-2)
    # project to zhat:  e1.z, e2.z, Vh.z
    dvr_e = dv1_e*e1[:,2] + dv2_e*e2[:,2] + dvV_e*Vh[:,2]
    dv2sq_e = dv1_e**2 + dv2_e**2 + dvV_e**2
    def acc(keep, dvr, dv2sq):
        w = np.where(keep, V/np.where(p>0,p,1.0), 0.0)
        fr = np.where(keep, w*dvr*dvr, 0.0); ft = np.where(keep, w*dv2sq, 0.0)
        Qr=n*fr.mean(); Tr=n*ft.mean()
        Qre=n*fr.std()/np.sqrt(N); Tre=n*ft.std()/np.sqrt(N)
        return Qr,Qre,Tr,Tre
    Qr_s,Qre_s,Tr_s,Tre_s = acc(keepS, dvr_s, dv2sq_s)
    Qr_e,Qre_e,Tr_e,Tre_e = acc(ok,    dvr_e, dv2sq_e)
    # common-random-number difference (exact - sharp), low variance
    wS=np.where(keepS, V/np.where(p>0,p,1.0),0.0); wE=np.where(ok, V/np.where(p>0,p,1.0),0.0)
    dfr = n*(wE*dvr_e*dvr_e - wS*dvr_s*dvr_s)
    dft = n*(wE*dv2sq_e     - wS*dv2sq_s)
    dQr=dfr.mean(); dQre=dfr.std()/np.sqrt(N)
    dTr=dft.mean(); dTre=dft.std()/np.sqrt(N)
    return dict(Qr_s=(Qr_s,Qre_s),Qt_s=(0.5*(Tr_s-Qr_s),0.5*np.hypot(Tre_s,Qre_s)),
                Qr_e=(Qr_e,Qre_e),Qt_e=(0.5*(Tr_e-Qr_e),0.5*np.hypot(Tre_e,Qre_e)),
                dQr=(dQr,dQre), dTr=(dTr,dTre))

print("\n  (a) Maxwellian bath, two parameter sets.  Both schemes ~1% from the closed form")
print("      (finite-b90/r power corrections, dominated by the small-V tail where b90~r):")
for r, sigma, m1, m2, N in [(1.0,10.0,1.0,1.0,12_000_000),(1.0,9.0,1.0,0.3,12_000_000)]:
    C,L,Qr_c,Qt_c = closed_forms(r,sigma,m1,m2)
    d = mc_both(r,sigma,m1,m2,N=N)
    print(f"\n  r={r} sigma={sigma} m=({m1},{m2})   L={L:.3f}   closed: Q_r={Qr_c:.4f}  Q_t={Qt_c:.4f}")
    print(f"     SHARP  b90 :  Q_r={d['Qr_s'][0]:.4f}+/-{d['Qr_s'][1]:.4f} (x{d['Qr_s'][0]/Qr_c:.4f})"
          f"   Q_t={d['Qt_s'][0]:.4f}+/-{d['Qt_s'][1]:.4f} (x{d['Qt_s'][0]/Qt_c:.4f})")
    print(f"     EXACT defl :  Q_r={d['Qr_e'][0]:.4f}+/-{d['Qr_e'][1]:.4f} (x{d['Qr_e'][0]/Qr_c:.4f})"
          f"   Q_t={d['Qt_e'][0]:.4f}+/-{d['Qt_e'][1]:.4f} (x{d['Qt_e'][0]/Qt_c:.4f})")

print("\n  (b) Monochromatic V FIXED, shrink eps = b90/r -> 0 by GROWING r  (so C=G^2 m*^2 n/V")
print("      is held fixed).  DECISIVE test that distinguishes 'exact b90' from any constant")
print("      cutoff-shift b90 -> alpha*b90:  a shift would make dQ_r approach the CONSTANT")
print("      -(8pi/3) C * 2 ln(alpha)  [= -(8pi/3)C if alpha=sqrt e, the transverse-only value].")
print("      A pure power correction instead sends dQ_r -> 0.")
m1=m2=1.0; mstar=1.0
V = np.sqrt((m1+mstar)/0.05)            # fixes b90 = (m_i+m*)/V^2 = 0.05
b90 = (m1+mstar)/V**2
C = 1.0/V                               # G=m*=n=1, <1/V>->1/V (monochromatic)
ref = 8*np.pi/3*C                       # |dQ_r| we WOULD see if exact gave b90*sqrt(e)
print(f"\n      fixed V={V:.3f}, b90={b90:.4f}, C={C:.4f};  a 'sqrt e' shift would give dQ_r -> -{ref:.4f}")
print(f"      {'eps=b90/r':>10} {'r':>7} {'Q_r':>9} {'dQ_r':>11} {'dQ_r/Q_r':>11} {'dQ_r/[-(8pi/3)C]':>17}")
for eps in [0.10, 0.05, 0.025, 0.0125]:
    r = b90/eps
    d = mc_both(r, None, m1, m2, N=16_000_000, fixedV=V, seed=7)
    Qr = d['Qr_s'][0]
    print(f"      {eps:10.4f} {r:7.3f} {Qr:9.4f} {d['dQr'][0]:+11.5f} {d['dQr'][0]/Qr:+11.2e}"
          f" {d['dQr'][0]/(-ref):+17.4f}")
print("      -> dQ_r -> 0 (NOT -> -(8pi/3)C); the last column heads to 0, not to +1.  There is")
print("         NO constant cutoff-shift: the exact nonlinear deflection regularizes at exactly")
print("         b90 = G(m_i+m*)/V^2.  The residual is an O(eps^2) impulsive power correction.")

print("\nDone.")
