#!/usr/bin/env python3
r"""
Fourier-space origin of the two Coulomb cutoffs b_{90,1}, b_{90,2} for point-mass perturbers.

This documents the SCHEME side of the regularization: how the sharp-b cutoff used by the
companion paper ("Evolution of Binaries Under Stochastic Perturbations", eqn:YpYp-pp)
produces the clean logarithm with no additive constant, and how an alternative (sharp-k)
regulator would shift it -- the shift that regularization-exactness.tex proves the exact
dynamics does NOT make (Lemma "Single-body self-energy"; see also the excision
shape-dependence discussed in its far-region lemma).

For a point mass tilde_rho(k)=1, the trace and anisotropy are
   Y_perp + Y_par = 4 pi int_0^inf (1 - J0(k rperp)) dk/k   (UV-divergent: close encounters),
   Y_perp - Y_par = 4 pi int_0^inf J2(k rperp) dk/k = 2 pi  (cutoff-free).
The divergent "1" is the SUM of the two bodies' self-energies. Regularizing each body's field by its
own window W_i(k) gives
   Y_perp + Y_par = 2 pi int [W1^2 + W2^2 - 2 W1 W2 J0(k rperp)] dk/k,
   Y_perp - Y_par = 4 pi int  W1 W2 J2(k rperp) dk/k.

CANONICAL SCHEME = sharp cut in b (standard Coulomb b_min): excise b_i<b_{90,i} from body i's field.
Its transform is W_i(k)=J0(k b_{90,i}), and the trace evaluates to the CLEAN log with NO constant:
   Y_perp + Y_par = 2 pi ln( rperp^2 / (b90_1 b90_2) ).            <- check (6)

ALTERNATIVE: a sharp cut in k, W_i=Theta(1/b_{90,i}-k), gives the same leading log + a
scheme-dependent constant 4 pi (gammaE - ln 2) [== rescaling b90_i -> 2 e^{-gammaE} b90_i].  <- check (5)
"""
import numpy as np
from scipy import integrate, special
gE = np.euler_gamma
PASS = lambda ok: "PASS" if ok else "**FAIL**"

# (1) asymptotic of the self integral: int_0^X (1-J0(t))/t dt - ln X -> gammaE - ln2
print("(1) int_0^X (1-J0)/t dt - ln X  ->  gammaE - ln2 = %.6f" % (gE-np.log(2)))
F = lambda X: integrate.quad(lambda t:(1-special.j0(t))/t, 0, X, limit=400)[0]
for X in (200.0, 1000.0, 5000.0):
    val = F(X) - np.log(X)
    print(f"    X={X:7.0f}:  {val:+.6f}   {PASS(abs(val-(gE-np.log(2)))<3e-3)}")

# (2) per-body split: sum_i int_0^{1/b90_i} (1-J0(k rp))/k dk  ==  ln(rp^2/(b1 b2)) + 2(gammaE-ln2)
print("\n(2) per-body split  vs  ln(rp^2/b1 b2) + 2(gammaE-ln2)   [bracket, i.e. /2pi]")
def split(rp, b1, b2):
    I = lambda K: integrate.quad(lambda k:(1-special.j0(k*rp))/k, 0, K, limit=600)[0]
    return I(1/b1) + I(1/b2)
for rp, b1, b2 in [(1.0,0.01,0.03), (1.0,0.005,0.02), (2.0,0.008,0.008)]:
    lhs = split(rp, b1, b2)
    rhs = np.log(rp**2/(b1*b2)) + 2*(gE-np.log(2))
    print(f"    rp={rp} b1={b1} b2={b2}:  split={lhs:+.5f}  closed={rhs:+.5f}  {PASS(abs(lhs-rhs)<5e-3)}")

# (3) anisotropy is cutoff-free: 4pi int_0^inf J2(k rp)/k dk = 2pi for any rp
print("\n(3) anisotropy 4pi int J2(k rp)/k dk = 2pi  (rp-independent, no cutoff)")
for rp in (0.5, 1.0, 3.0):
    val = 4*np.pi*integrate.quad(lambda k: special.jv(2,k*rp)/k, 1e-9, np.inf, limit=800)[0]
    print(f"    rp={rp}:  {val:.5f}  (2pi={2*np.pi:.5f})  {PASS(abs(val-2*np.pi)<5e-3)}")

# (4) scheme conversion: sharp k-cut at 1/b90 == sharp b-cut at (2 e^{-gammaE}) b90
print("\n(4) sharp-k cut at 1/b90 matches sharp-b cut at  2 e^{-gammaE} b90 = %.4f b90" % (2*np.exp(-gE)))

# (5) ALTERNATIVE sharp-k window: tilde_y = 2pi i rho~ (k/k^2)[W1 - W2 e^{-ik.rp}], Wi=Theta(1/b90_i-k)
#     => Yperp+Ypar = 2pi int [W1^2+W2^2-2 W1 W2 J0]dk/k = 2pi ln(rp^2/b1 b2) + 4pi(gE-ln2)
print("\n(5) sharp-k window W_i=Theta(1/b90_i-k): trace = clean log + 4pi(gE-ln2)")
def trace_sharpk(rp,b1,b2):
    K1,K2=1/b1,1/b2
    integ=lambda k:((1.0 if k<K1 else 0.0)+(1.0 if k<K2 else 0.0)
                    -2*(1.0 if k<min(K1,K2) else 0.0)*special.j0(k*rp))/k
    return 2*np.pi*integrate.quad(integ,0,max(K1,K2),limit=800,points=[min(K1,K2)])[0]
def aniso_sharpk(rp,b1,b2):
    return 4*np.pi*integrate.quad(lambda k: special.jv(2,k*rp)/k,0,min(1/b1,1/b2),limit=800)[0]
for rp,b1,b2 in [(1.0,0.01,0.03),(1.0,0.005,0.02),(2.0,0.008,0.015)]:
    tr=trace_sharpk(rp,b1,b2); trc=2*np.pi*np.log(rp**2/(b1*b2))+4*np.pi*(gE-np.log(2))
    an=aniso_sharpk(rp,b1,b2)
    print(f"    rp={rp} b1={b1} b2={b2}:  trace={tr:.4f} vs {trc:.4f} {PASS(abs(tr-trc)<0.05)}"
          f" | aniso={an:.4f} vs 2pi={2*np.pi:.4f} {PASS(abs(an-2*np.pi)<0.05)}")

# (6) CANONICAL sharp-b window W_i=J0(k b90_i)  (FT of point-mass field with disk b_i<b90_i removed)
#     => Yperp+Ypar = 2pi int [J0(k b1)^2+J0(k b2)^2 - 2 J0(k b1)J0(k b2)J0(k rp)] dk/k
#                   = 2pi ln(rp^2/b1 b2)   WITH NO CONSTANT (clean Coulomb log).
# The integrand decays only as 1/k^2 with O(1) oscillations, so adaptive quad is unreliable;
# use a fine fixed grid (vectorized) + the analytic non-oscillatory tail
#   int_K^inf <J0(kb1)^2+J0(kb2)^2>/k dk = (1/(pi K))(1/b1+1/b2)   (<J0^2>=1/(pi k b)).
print("\n(6) CANONICAL sharp-b window W_i=J0(k b90_i): trace = clean log (NO constant)")
def trace_sharpb(rp,b1,b2,K=1.0e5,dk=0.01):
    k=np.arange(dk,K,dk)
    h=(special.j0(k*b1)**2+special.j0(k*b2)**2
       -2*special.j0(k*b1)*special.j0(k*b2)*special.j0(k*rp))/k
    tail=(1.0/(np.pi*K))*(1.0/b1+1.0/b2)
    return 2*np.pi*(np.trapz(h,dx=dk)+tail)
def aniso_sharpb(rp,b1,b2,K=2.0e4,dk=0.01):     # integrand ~k^{-5/2}, converges fast
    k=np.arange(dk,K,dk)
    h=special.j0(k*b1)*special.j0(k*b2)*special.jv(2,k*rp)/k
    return 4*np.pi*np.trapz(h,dx=dk)
for rp,b1,b2 in [(1.0,0.02,0.05),(2.0,0.03,0.08),(1.0,0.01,0.04)]:
    tr=trace_sharpb(rp,b1,b2); trc=2*np.pi*np.log(rp**2/(b1*b2))   # NO 4pi(gE-ln2)
    an=aniso_sharpb(rp,b1,b2)
    print(f"    rp={rp} b1={b1} b2={b2}:  trace={tr:.4f} vs {trc:.4f} {PASS(abs(tr-trc)<0.05)}"
          f" | aniso={an:.4f} vs 2pi={2*np.pi:.4f} {PASS(abs(an-2*np.pi)<0.05)}")
