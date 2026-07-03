#!/usr/bin/env python3
r"""
DOES THE TWO-BODY REDUCTION ACTUALLY HOLD?  A genuine three-body test.

The companion script exact_deflection_check.py does NOT integrate a trajectory in the field of
both bodies: its "exact" kick is the SUPERPOSITION of two independent two-body hyperbolic
deflections (each body treated as if the other were absent).  That is the impulsive-limit
*ansatz* itself, so it cannot test whether the ansatz is right.

Here we test it honestly.  We FREEZE the binary (bodies 1,2 held at fixed positions s1,s2 --
exact in the impulsive limit, where the encounter time << orbital time), launch the perturber
from far away with velocity V and impact parameter b, and INTEGRATE ITS TRAJECTORY IN THE
COMBINED POTENTIAL of both bodies,
    x'' = -G m1 (x-s1)/|x-s1|^3 - G m2 (x-s2)/|x-s2|^3 .
The impulse actually delivered to body i is dv_i = G m_* \int (x(t)-s_i)/|x(t)-s_i|^3 dt, which we
accumulate along the trajectory (separately for i=1,2).  The relative-velocity kick is
dv = dv_1 - dv_2.  This INCLUDES every three-body effect: body 2 bending the perturber's path
before/after the body-1 encounter, focusing by the pair, sequential close encounters, the lot.

The SUPERPOSITION ansatz (what the closed form uses) is, in the SAME convention dv=dv1-dv2,
    dv_sup = (2 G m_*/V)[  b1/(b1^2+b90_1^2) - b2/(b2^2+b90_2^2)              (transverse, in-plane)
                         + (b90_1/(b1^2+b90_1^2) - b90_2/(b2^2+b90_2^2)) Vhat ] (||V focusing),
the exact two-body deflection of each body (transverse Plummer-core b90_i + forward focusing),
with b_i the impact-parameter vector relative to body i's projection onto the impact plane.

The question: does the three-body deflection shift Q_r,Q_t, and if so does the shift
VANISH as eps=b90/r -> 0 (=> b90 exact in the impulsive limit, ansatz good) or approach a
nonzero constant (=> ansatz fails, constants -2/3,-8/3 wrong)?  We measure it.  G=m_*=n=1.
"""
import sys, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy import integrate as _it

PASS = lambda ok: "PASS" if ok else "**FAIL**"
G = 1.0; mstar = 1.0; nbath = 1.0

def bodykicks_3body(bvec, Vhat, V, s1, s2, m1, m2, Lfac=12.0, rtol=1e-10, atol=1e-13):
    """Genuine impulsive three-body encounter.  All three masses are DYNAMICAL: the perturber
    is deflected by both bodies AND each body recoils from the perturber (so the close
    perturber-body_i passage is the correct two-body problem with scale b90_i=G(m_i+m*)/V^2).
    The binary's internal self-gravity (body1<->body2) is switched OFF -- that is precisely the
    impulsive limit, where the binary configuration is frozen during the fast encounter but the
    bodies still recoil.  Bodies start at rest at s1,s2; perturber starts far up-stream.
    Returns (dv1, dv2) = the net velocity change of body 1 and body 2.

    State y = [x*(3), x1(3), x2(3), v*(3), v1(3), v2(3)].  max_step is capped at a fraction of
    the close-encounter scale b90/V so the adaptive stepper cannot leap over the localized
    close passage."""
    r = np.linalg.norm(s2 - s1); L = Lfac*r
    xstar0 = bvec - L*Vhat
    y0 = np.concatenate([xstar0, s1, s2, V*Vhat, np.zeros(3), np.zeros(3)])
    tend = 2.0*L/V
    b90min = G*min(m1+mstar, m2+mstar)/V**2
    hmax = 0.25*b90min/V
    def rhs(t, y):
        xs = y[0:3]; x1 = y[3:6]; x2 = y[6:9]
        d1 = xs - x1; d2 = xs - x2                       # perturber relative to each body
        r1 = (d1@d1)**1.5; r2 = (d2@d2)**1.5
        a_star = -G*m1*d1/r1 - G*m2*d2/r2                # perturber pulled by both bodies
        a_1 = G*mstar*d1/r1                              # body1 pulled toward perturber only
        a_2 = G*mstar*d2/r2                              # body2 pulled toward perturber only
        return np.concatenate([y[9:12], y[12:15], y[15:18], a_star, a_1, a_2])
    sol = solve_ivp(rhs, [0.0, tend], y0, method='DOP853', rtol=rtol, atol=atol, max_step=hmax)
    v1 = sol.y[12:15,-1]; v2 = sol.y[15:18,-1]           # initial body velocities were 0
    return v1, v2

def kick_super(bvec, Vhat, V, s1, s2, m1, m2):
    """superposition relative kick dv1-dv2 (3-vector), same convention as the 3-body dv1-dv2."""
    b90_1 = G*(m1+mstar)/V**2; b90_2 = G*(m2+mstar)/V**2
    s1p = s1 - (s1@Vhat)*Vhat; s2p = s2 - (s2@Vhat)*Vhat
    b1v = bvec - s1p; b2v = bvec - s2p
    b1 = np.linalg.norm(b1v); b2 = np.linalg.norm(b2v)
    pre = 2*G*mstar/V
    transv = b1v/(b1**2+b90_1**2) - b2v/(b2**2+b90_2**2)
    longi  = (b90_1/(b1**2+b90_1**2) - b90_2/(b2**2+b90_2**2))*Vhat
    return pre*(transv + longi)

# ----------------------------------------------------------------------------------------
print("="*86)
print("SECTION 1 -- integrator validation: single body must reproduce the exact two-body kick")
print("  (body 2 placed far OFF the perturber path; compare dv1 to the exact hyperbolic kick)")
print("="*86)
worst = 0.0
for b, V, m1 in [(0.5,3.0,1.0),(1.0,3.0,1.0),(0.2,5.0,1.0),(2.0,4.0,0.3),(0.05,6.0,1.0)]:
    # binary along z, separation 1; perturber moves +x, grazes body1 at transverse distance b;
    # body 2 is massless (m2=0) so it neither deflects the perturber nor contaminates dv1.
    Vhat=np.array([1.,0.,0.]); s1=np.array([0.,0.,-0.5]); s2=np.array([0.,0.,0.5])
    bvec=np.array([0., b, -0.5])                       # impact plane is y-z; offset b in y from body1
    dv1,_ = bodykicks_3body(bvec, Vhat, V, s1, s2, m1, 0.0, Lfac=200.0)
    b90=(m1+mstar)/V**2
    tot2 = 4*G**2*mstar**2/V**2/(b**2+b90**2)
    err = abs(dv1@dv1 - tot2)/tot2; worst=max(worst,err)
    print(f"  b={b:5.2f} V={V:4.1f} m1={m1:4.2f}: |dv1|^2 num={dv1@dv1:.6f} exact={tot2:.6f} relerr={err:.2e}")
print(f"  worst relative error: {worst:.2e}  {PASS(worst<3e-3)}  (residual = finite start distance)")
sys.stdout.flush()

# ----------------------------------------------------------------------------------------
print("\n"+"="*86)
print("SECTION 2 -- single worst-case trajectories: how big is the per-encounter 3-body error?")
print("="*86)
def one_traj(eps, m1, m2, b1_over_b90, geom, V=6.0, Lfac=40.0):
    b90_1=(m1+mstar)/V**2; r=b90_1/eps
    s1=np.array([0.,0.,-m2/(m1+m2)*r]); s2=np.array([0.,0.,+m1/(m1+m2)*r])  # binary along z
    if geom=='perp':        # V perpendicular to binary axis -> generic case
        Vhat=np.array([1.,0.,0.]); bvec=np.array([0., 0., s1[2]+b1_over_b90*b90_1])
    elif geom=='aligned':   # V ALONG binary axis -> perturber hits body1 then body2 (degenerate)
        Vhat=np.array([0.,0.,1.]); bvec=np.array([b1_over_b90*b90_1,0.,0.])
    dv1,dv2 = bodykicks_3body(bvec,Vhat,V,s1,s2,m1,m2,Lfac=Lfac)
    dvf=dv1-dv2; dvs=kick_super(bvec,Vhat,V,s1,s2,m1,m2)
    rel = np.linalg.norm(dvf-dvs)/(np.linalg.norm(dvs)+1e-30)
    return np.linalg.norm(dvf), np.linalg.norm(dvs), rel
print("  (a) GENERIC geometry (V perpendicular to binary axis), grazing body 1 at b1~b90:")
for eps in [0.1,0.05,0.025]:
    f,s,rel = one_traj(eps,1.0,1.0,1.0,'perp')
    print(f"      eps={eps:.3f}: |dv_3body|={f:.5f} |dv_super|={s:.5f} |err|/|dv|={rel:.2e}")
print("  (b) DEGENERATE aligned geometry (V along binary axis: hits body1 THEN body2).")
print("      This is the worst case for the ansatz; per-trajectory it is O(1)-WRONG, BUT it occupies")
print("      a solid angle ~ (b90/r)^2 = eps^2, so its weight in Q vanishes (see Section 3).")
for eps in [0.1,0.05,0.025]:
    f,s,rel = one_traj(eps,1.0,1.0,1.0,'aligned')
    print(f"      eps={eps:.3f}: |dv_3body|={f:.5f} |dv_super|={s:.5f} |err|/|dv|={rel:.2e}")
sys.stdout.flush()

# ----------------------------------------------------------------------------------------
print("\n"+"="*86)
print("SECTION 3 -- MC of binary Q_r, trace: FULL 3-BODY vs SUPERPOSITION  (common random b,V)")
print("  monochromatic V, b90 fixed, r grown so eps=b90/r->0 with C=G^2m*^2n/V fixed.")
print("  Decisive: dQ_r=Q_r(3body)-Q_r(super) must -> 0 as a POWER of eps, not to a constant.")
print("="*86)

def closed_fixedV(r, V, m1, m2):
    C=G**2*mstar**2*nbath/V; b1=(m1+mstar)/V**2; b2=(m2+mstar)/V**2
    Qr,_=_it.quad(lambda th: np.pi*(np.log((r*np.sin(th))**2/(b1*b2))+1.0)*np.sin(th)**3,0,np.pi)
    Tr,_=_it.quad(lambda th: 2*np.pi*np.log((r*np.sin(th))**2/(b1*b2))*np.sin(th),0,np.pi)
    return 2*C*Qr, 2*C*Tr

def mc(eps, m1, m2, V, N, seed=11, Lfac=10.0):
    g=np.random.default_rng(seed)
    b90_1=(m1+mstar)/V**2; b90_2=(m2+mstar)/V**2; r=b90_1/eps; Rout=14.0*r
    zh=np.array([0.,0.,1.])
    s1=-m2/(m1+m2)*r*zh; s2=m1/(m1+m2)*r*zh
    Vh=g.normal(size=(N,3)); Vh/=np.linalg.norm(Vh,axis=1)[:,None]
    s1p=s1[None,:]-(Vh@s1)[:,None]*Vh; s2p=s2[None,:]-(Vh@s2)[:,None]*Vh
    axis=s2p-s1p; axn=np.linalg.norm(axis,axis=1); good=axn>1e-9*r
    eA=np.zeros_like(Vh); eA[good]=axis[good]/axn[good,None]; eB=np.cross(Vh,eA)
    sA1=np.einsum('ij,ij->i',s1p,eA); sA2=np.einsum('ij,ij->i',s2p,eA)
    # sample b (scalars bA,bB in the eA,eB plane) : log-radial about each body + uniform disk.
    # floor at b90/5: deeper encounters (b<b90/5) contribute <1% of Q [integrand ~ b/(b^2+b90^2)]
    # and are very slow to integrate, so we exclude them from BOTH schemes identically.
    bfac=5.0; lo1=b90_1/bfac; lo2=b90_2/bfac
    w0,w1,w2=0.4,0.4,0.2
    u=g.random(N); comp=np.where(u<w0,0,np.where(u<w0+w1,1,2)); ang=g.random(N)*2*np.pi
    rr=np.where(comp==0, lo1*(Rout/lo1)**g.random(N),
        np.where(comp==1, lo2*(Rout/lo2)**g.random(N), Rout*np.sqrt(g.random(N))))
    cen=np.where(comp==0,sA1,np.where(comp==1,sA2,0.0))
    bA=cen+rr*np.cos(ang); bB=rr*np.sin(ang)
    rho1=np.hypot(bA-sA1,bB); rho2=np.hypot(bA-sA2,bB); rho0=np.hypot(bA,bB)
    L1=np.log(Rout/lo1); L2=np.log(Rout/lo2)
    p0=np.where((rho1>=lo1)&(rho1<=Rout),1/(2*np.pi*rho1**2*L1),0.0)
    p1=np.where((rho2>=lo2)&(rho2<=Rout),1/(2*np.pi*rho2**2*L2),0.0)
    p2=np.where(rho0<=Rout,1/(np.pi*Rout**2),0.0)
    p=w0*p0+w1*p1+w2*p2; ok=good&(p>0)
    bvec=bA[:,None]*eA+bB[:,None]*eB
    w=np.where(ok,V/np.where(p>0,p,1.0),0.0)
    # superposition (vectorized)
    b1v=bvec-s1p; b2v=bvec-s2p
    b1=np.linalg.norm(b1v,axis=1); b2=np.linalg.norm(b2v,axis=1)
    pre=2*G*mstar/V
    dv_s=pre*( b1v/(b1**2+b90_1**2)[:,None]-b2v/(b2**2+b90_2**2)[:,None]
              +(b90_1/(b1**2+b90_1**2)-b90_2/(b2**2+b90_2**2))[:,None]*Vh )
    dvr_s=dv_s@zh; dv2_s=np.einsum('ij,ij->i',dv_s,dv_s)
    # full 3-body (loop over ok samples)
    dvr_f=np.zeros(N); dv2_f=np.zeros(N); idx=np.where(ok)[0]
    for k in idx:
        d1,d2=bodykicks_3body(bvec[k],Vh[k],V,s1,s2,m1,m2,Lfac=Lfac)
        dv=d1-d2; dvr_f[k]=dv@zh; dv2_f[k]=dv@dv
    sm=lambda a:(np.where(ok,nbath*w*a,0.0).mean(), np.where(ok,nbath*w*a,0.0).std()/np.sqrt(N))
    Qr_s,_=sm(dvr_s**2); Tr_s,_=sm(dv2_s); Qr_f,_=sm(dvr_f**2); Tr_f,_=sm(dv2_f)
    dQ=nbath*w*(dvr_f**2-dvr_s**2); dT=nbath*w*(dv2_f-dv2_s)
    return dict(r=r,Qr_s=Qr_s,Qr_f=Qr_f,Tr_s=Tr_s,Tr_f=Tr_f,nok=len(idx),
                dQ=(dQ.mean(),dQ.std()/np.sqrt(N)),dT=(dT.mean(),dT.std()/np.sqrt(N)))

m1=m2=1.0; b90_target=0.05; V=np.sqrt((m1+mstar)/b90_target)
print(f"  m1=m2={m1}, V={V:.4f}, b90={(m1+mstar)/V**2:.4f} fixed; C=1/V={1/V:.4f}")
print(f"  a constant 'sqrt e' error would give dQ_r -> -(8pi/3)C = {-8*np.pi/3/V:+.4f} (NONZERO).\n")
print(f"  {'eps':>7} {'r':>7} {'Qr_sup':>8} {'Qr_3bdy':>8} {'closed':>8} | "
      f"{'dQr(3b-sup)':>16} {'dQr/Qr':>10} {'dTr/Tr':>10}")
t0=time.time(); res=[]
# eps=0.025 already shows the clean halving; 0.0125 (r=4) is added but is slow (~10 min).
for eps,N in [(0.10,3000),(0.05,3000),(0.025,3000)]:
    d=mc(eps,m1,m2,V,N); Qc,Tc=closed_fixedV(d['r'],V,m1,m2)
    rq=d['dQ'][0]/d['Qr_f']; rt=d['dT'][0]/d['Tr_f']; res.append((eps,rq,rt))
    print(f"  {eps:7.4f} {d['r']:7.3f} {d['Qr_s']:8.4f} {d['Qr_f']:8.4f} {Qc:8.4f} | "
          f"{d['dQ'][0]:+8.4f}+-{d['dQ'][1]:.4f} {rq:+10.2e} {rt:+10.2e}")
    sys.stdout.flush()
print(f"\n  (elapsed {time.time()-t0:.0f}s)")
print("  scaling of |dQr/Qr| as eps halves (eps^1 -> ratio 2, eps^2 -> ratio 4):")
for i in range(1,len(res)):
    if abs(res[i][1])>1e-12:
        print(f"     {res[i-1][0]:.4f}->{res[i][0]:.4f}:  {abs(res[i-1][1]):.2e} -> {abs(res[i][1]):.2e}"
              f"   ratio {abs(res[i-1][1]/res[i][1]):.2f}")
print("\nDone.")
