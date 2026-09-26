"""
================================================================================
PRICER D'OPTIONS — Black-Scholes, Greeks (1er + 2e ordre), calibration marche
================================================================================
Projet vers MIngFin / entretiens quant.

  STAGE 1 : prix call/put + les cinq greeks de premier ordre (forme spot)
  STAGE 2 : forme "forward" (Black-76) + extraction du forward par parite
  STAGE 3 : greeks de second ordre — vanna & volga

Entrees (forme spot) :
    S=spot, K=strike, T=temps en ANNEES, r=taux (decimal), sigma=vol (decimal)
================================================================================
"""

import numpy as np
from scipy.stats import norm


# ==============================================================================
# STAGE 1 — Black-Scholes spot + 5 greeks
# ==============================================================================
def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_call_price(S, K, T, r, sigma):
    d1, d2 = _d1(S, K, T, r, sigma), _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    d1, d2 = _d1(S, K, T, r, sigma), _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def delta_call(S, K, T, r, sigma):
    return norm.cdf(_d1(S, K, T, r, sigma))

def delta_put(S, K, T, r, sigma):
    return norm.cdf(_d1(S, K, T, r, sigma)) - 1.0

def gamma(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    # par +1 point de % de vol  ->  /100
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0

def theta_call(S, K, T, r, sigma):
    d1, d2 = _d1(S, K, T, r, sigma), _d2(S, K, T, r, sigma)
    t1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    t2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    return (t1 + t2) / 365.0   # par jour

def theta_put(S, K, T, r, sigma):
    d1, d2 = _d1(S, K, T, r, sigma), _d2(S, K, T, r, sigma)
    t1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    t2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return (t1 + t2) / 365.0

def rho_call(S, K, T, r, sigma):
    d2 = _d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0

def rho_put(S, K, T, r, sigma):
    d2 = _d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


# ==============================================================================
# STAGE 2 — forme forward (sans r ni q devines) + extraction par parite
# ==============================================================================
def _d1_fwd(F, K, T, sigma):
    return (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))

def bs_call_fwd(F, K, T, DF, sigma):
    d1 = _d1_fwd(F, K, T, sigma); d2 = d1 - sigma * np.sqrt(T)
    return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))

def bs_put_fwd(F, K, T, DF, sigma):
    d1 = _d1_fwd(F, K, T, sigma); d2 = d1 - sigma * np.sqrt(T)
    return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def forward_from_parity(strikes, call_mids, put_mids):
    """C - P = DF*(F - K)  ->  droite : pente=-DF, ordonnee=DF*F. Rend (F, DF)."""
    K = np.asarray(strikes, float)
    CmP = np.asarray(call_mids, float) - np.asarray(put_mids, float)
    slope, intercept = np.polyfit(K, CmP, 1)
    DF = -slope
    return intercept / DF, DF


# ==============================================================================
# STAGE 3 — greeks de second ordre : vanna & volga
# ==============================================================================
# VANNA = d(delta)/d(sigma) = d(vega)/d(S) = d2V / dS dsigma
#   Le pont spot <-> vol. Derriere le SKEW. Nulle a l'ATM, pics dans les ailes.
#   Formule (q=0) :  vanna = -phi(d1) * d2 / sigma
#   Convention ici : par +1% de vol  ->  /100  (comme le vega).
def vanna(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma); d2 = d1 - sigma * np.sqrt(T)
    return -norm.pdf(d1) * d2 / sigma / 100.0

# VOLGA (vomma) = d(vega)/d(sigma) = d2V / dsigma^2
#   La convexite du vega a la vol. Quasi nulle a l'ATM, pics dans les ailes.
#   -> pour parier sur la "vol de vol", on achete les ailes, pas l'ATM.
#   Formule (q=0) :  volga = vega_brut * d1*d2 / sigma
#   Convention ici : variation du vega (deja en /100) par +1% de vol.
def volga(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma); d2 = d1 - sigma * np.sqrt(T)
    vega_brut = S * norm.pdf(d1) * np.sqrt(T)
    return vega_brut * d1 * d2 / sigma / 100.0 / 100.0


# ==============================================================================
# DEMO — donnees reelles Nov 3 (38j), tout extrait des prix
# ==============================================================================
if __name__ == "__main__":
    T = 38 / 365
    strikes   = [7650,   7675,   7700,   7725,   7750  ]
    call_mids = [205.20, 187.35, 170.25, 153.90, 138.55]
    put_mids  = [ 82.60,  89.75,  97.55, 106.10, 115.45]

    F, DF = forward_from_parity(strikes, call_mids, put_mids)
    r = -np.log(DF) / T
    print(f"Forward extrait des prix : F={F:.1f}  DF={DF:.4f}  r={r:.2%}")

    call = bs_call_fwd(F, 7750, T, DF, 0.1264)
    print(f"Call 7750 (bid 137.40)   : {call:.2f}  vs Moomoo mid 138.55")

    # Greeks de second ordre au strike 7750 (forme spot, S~forward, r extrait)
    S = float(F)
    print(f"Vanna 7750 : {vanna(S,7750,T,r,0.1264):+.5f}  (dDelta par +1% de vol)")
    print(f"Volga 7750 : {volga(S,7750,T,r,0.1264):+.5f}  (dVega  par +1% de vol)")