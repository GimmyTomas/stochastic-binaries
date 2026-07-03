"""Diffusion-time calculators for the astrophysical applications of Sec. V.

All functions return T_d in Gyr. Internal units: pc, km/s, M_sun
(1 time unit = pc/(km/s) = 0.977792 Myr), except the ULDM formula which is
evaluated in SI. Each function implements a labeled equation of the paper:

    td_uldm                 eqn:Td-uldm     (slow ULDM density fluctuations)
    td_tidal_perturbers     eqn:Td-tidal-R  = eqn:Td-large (extended, R > a)
    td_pointmass_perturbers eqn:Td* with T_d ~ T_d^*/logLambda
                            (e-dependent factors neglected, as in
                            eqn:Td-small-gaia)
    td_ism                  the Gamma-function formula of sec:ism
    td_gw                   sec:blas (GW background with Omega_gw ~ 1/f)
    td_from_Prho            eqn:Td-Prho (generic density power spectrum,
                            numeric double integral)
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.special import gamma as Gamma

G_PC = 4.300917270e-3          # G in pc (km/s)^2 / Msun
UNIT_MYR = 0.977792            # pc/(km/s) in Myr
UNIT_GYR = UNIT_MYR / 1000.0   # pc/(km/s) in Gyr

# SI constants (for the ULDM formula)
G_SI = 6.67430e-11
HBAR = 1.054571817e-34
EV_KG = 1.782661922e-36        # 1 eV/c^2 in kg
PC_M = 3.0856775814913673e16
MSUN_KG = 1.98841e30
SEC_GYR = 1 / 3.15576e16


def td_uldm(m_msun=1.0, mu_ev=1e-21, sigma_kms=100.0, rho_msun_pc3=1.0, a_pc=0.1):
    """eqn:Td-uldm: T_d = m mu sigma^2 / (32 pi^2 G rho^2 a^3 hbar)."""
    m = m_msun * MSUN_KG
    mu = mu_ev * EV_KG
    sig = sigma_kms * 1e3
    rho = rho_msun_pc3 * MSUN_KG / PC_M**3
    a = a_pc * PC_M
    td_s = m * mu * sig**2 / (32 * np.pi**2 * G_SI * rho**2 * a**3 * HBAR)
    return td_s * SEC_GYR


def td_tidal_perturbers(m_msun=1.0, mstar_msun=1e3, sigma_kms=100.0,
                        R_pc=0.1, rho_msun_pc3=0.026, a_pc=0.1):
    """eqn:Td-tidal-R / eqn:Td-large: T_d = sigma R^2 m/(4 sqrt(2pi) G m_*^2 n a^3)."""
    n = rho_msun_pc3 / mstar_msun
    td = (sigma_kms * R_pc**2 * m_msun
          / (4 * np.sqrt(2 * np.pi) * G_PC * mstar_msun**2 * n * a_pc**3))
    return td * UNIT_GYR


def log_lambda_pointmass(sigma_kms, a_pc, m1_msun, m2_msun, mstar_msun, e=0.0):
    """eqn:logLambda (pointlike perturbers)."""
    w = np.sqrt(1 - e**2)
    return (np.log(4 * sigma_kms**4 * a_pc**2 * (1 + w) ** 2
                   / (G_PC**2 * (m1_msun + mstar_msun) * (m2_msun + mstar_msun)))
            - 2 * 0.5772156649015329)


def td_pointmass_perturbers(m_msun=1.0, mstar_msun=1.0, sigma_kms=100.0,
                            rho_msun_pc3=0.026, a_pc=0.1, logLam=None):
    """T_d ~ T_d^*/logLambda (eqn:Td*, e-dependent factors neglected as in
    eqn:Td-small-gaia). T_d^* = m sigma/(40 sqrt(2pi) G m_*^2 n a)."""
    n = rho_msun_pc3 / mstar_msun
    if logLam is None:
        logLam = log_lambda_pointmass(sigma_kms, a_pc, m_msun / 2, m_msun / 2,
                                      mstar_msun)
    tds = m_msun * sigma_kms / (40 * np.sqrt(2 * np.pi) * G_PC
                                * mstar_msun**2 * n * a_pc)
    return tds / logLam * UNIT_GYR


ISM_MODELS = {
    # sec:ism (TIGRESS-NCR parameters of Modak et al.)
    "R8": dict(Sigma=12.0, L=1024.0, P0=2.4, n=2.3, tau0_Myr=5.0, veff=12.0, D=354.0),
    "LGR4": dict(Sigma=50.0, L=512.0, P0=1.7, n=2.2, tau0_Myr=5.0, veff=10.0, D=363.0),
}


def td_ism(m_msun=1.0, a_pc=0.1, model="R8"):
    """sec:ism: T_d = [2 sqrt(pi)/(Gamma((3-n)/2) Gamma((n-2)/2))]
    * m D v_eff^{3-n} tau_0^{2-n} / (32 G a^3 Sigma^2 L^{2-n} P_d0)."""
    p = ISM_MODELS[model]
    nn = p["n"]
    tau0 = p["tau0_Myr"] / UNIT_MYR   # in pc/(km/s)
    pref = 2 * np.sqrt(np.pi) / (Gamma((3 - nn) / 2) * Gamma((nn - 2) / 2))
    td = pref * (m_msun * p["D"] * p["veff"] ** (3 - nn) * tau0 ** (2 - nn)
                 / (32 * G_PC * a_pc**3 * p["Sigma"] ** 2
                    * p["L"] ** (2 - nn) * p["P0"]))
    return td * UNIT_GYR


def td_gw(m_msun=1.0, a_pc=0.1, f_hz=1e-8, omega_gw=1e-9, h0_kms_mpc=70.0):
    """sec:blas: T_d = Gm/(9 pi^2 a^3 H0^2 f Omega_gw(f)) for Omega_gw ~ 1/f."""
    H0 = h0_kms_mpc / 3.0857e19            # 1/s
    m = m_msun * MSUN_KG
    a = a_pc * PC_M
    td_s = G_SI * m / (9 * np.pi**2 * a**3 * H0**2 * f_hz * omega_gw)
    return td_s * SEC_GYR


def td_from_Prho(P_rho, m_msun=1.0, a_pc=0.1, p_max=np.inf, tau_max=np.inf):
    """eqn:Td-Prho: T_d = Gm/((4 pi G)^2 a^3) / [Int dtau Int p^2 dp/(2 pi^2) P_rho].

    ``P_rho(p, tau)`` in units of Msun^2 pc^-3 (p in 1/pc, tau in pc/(km/s)).
    """
    inner, _ = dblquad(lambda tau, p: p**2 / (2 * np.pi**2) * 2 * P_rho(p, tau),
                       0, p_max, 0, tau_max)  # factor 2: tau integral is even
    td = G_PC * m_msun / ((4 * np.pi * G_PC) ** 2 * a_pc**3) / inner
    return td * UNIT_GYR


def _selfcheck():
    """td_from_Prho must reproduce the closed Gaussian-bath form eqn:Td-tidal-R."""
    mstar, sig, R, rho = 1e3, 200.0, 0.5, 0.0104
    n = rho / mstar

    def P_rho(p, tau):
        return mstar**2 * n * np.exp(-p**2 * R**2) * np.exp(-p**2 * sig**2 * tau**2 / 2)

    got = td_from_Prho(P_rho, m_msun=1.0, a_pc=0.1)
    want = td_tidal_perturbers(1.0, mstar, sig, R, rho, 0.1)
    rel = abs(got / want - 1)
    ok = rel < 1e-5  # dblquad default accuracy
    print(f"td_from_Prho vs closed form: rel diff = {rel:.2e} "
          f"({'PASS' if ok else 'FAIL'})")
    return ok


if __name__ == "__main__":
    _selfcheck()
    print(f"ULDM     T_d = {td_uldm():8.2f} Gyr   (paper: 3.8 Gyr at reference values)")
    print(f"subhalo  T_d = {td_tidal_perturbers():8.3f} Gyr (paper: 0.87 Gyr)")
    print(f"pt-mass  T_d = {td_pointmass_perturbers(logLam=25):8.2f} Gyr  (paper: 3.5 Gyr)")
    print(f"ISM R8   T_d = {td_ism(model='R8'):8.1f} Gyr  (paper: 45 Gyr)")
    print(f"ISM LGR4 T_d = {td_ism(model='LGR4'):8.2f} Gyr  (paper: 1.6 Gyr)")
