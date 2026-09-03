"""
================================================================================
STAGE 1 — Black-Scholes pricer + les cinq Greeks
================================================================================

Projet : pricer d'options en Python (vers MIngFin / entretiens quant).

Ce module fait deux choses :
  1. Prix d'un call et d'un put europeens (formule de Black-Scholes).
  2. Les cinq Greeks : delta, gamma, theta, vega, rho.

Rappel des cinq entrees (Day 3) :
    S     = prix spot du sous-jacent aujourd'hui
    K     = strike (prix d'exercice)
    T     = temps jusqu'a l'echeance, EN ANNEES (ex. 3 mois = 0.25)
    r     = taux sans risque (decimal : 5% = 0.05)
    sigma = volatilite (decimal : 20% = 0.20)

Convention : les taux et la volatilite sont annualises.
================================================================================
"""

import numpy as np
from scipy.stats import norm   # norm.cdf = N(.), norm.pdf = densite normale


# ------------------------------------------------------------------------------
# 1. Les briques de base : d1 et d2
# ------------------------------------------------------------------------------
# La formule de Black-Scholes repose sur deux quantites, d1 et d2.
# Tu n'as pas a les memoriser : tu les codes une fois, et tout en decoule.
#
#     d1 = [ ln(S/K) + (r + sigma^2 / 2) * T ] / ( sigma * sqrt(T) )
#     d2 = d1 - sigma * sqrt(T)
#
# Le terme sigma*sqrt(T) est la "quantite de mouvement" possible d'ici l'echeance
# (Day 3). Il revient partout, d'ou une petite fonction dediee.

def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


# ------------------------------------------------------------------------------
# 2. Prix des options
# ------------------------------------------------------------------------------
# Call : C = S * N(d1)  -  K * e^(-rT) * N(d2)
# Put  : P = K * e^(-rT) * N(-d2)  -  S * N(-d1)
#
# Structure a reconnaitre (Day 3) :
#   - S*N(d1)         = valeur attendue de recevoir l'action, ponderee proba
#   - K*e^(-rT)*N(d2) = valeur attendue de payer le strike, ACTUALISEE
#   - la soustraction = "ce que je recois moins ce que je paie", en esperance
#   - N(d2)           = proba risque-neutre de finir dans la monnaie
#   - e^(-rT)         = facteur d'actualisation

def bs_call_price(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ------------------------------------------------------------------------------
# 3. Les cinq Greeks (Day 4)
# ------------------------------------------------------------------------------
# Chaque Greek = une derivee du prix par rapport a une entree.
# C'est en les codant qu'on les comprend, donc chacun est commente.

# DELTA — sensibilite du prix a une variation de $1 du sous-jacent.
#   Call : entre 0 et 1.  Put : entre -1 et 0.
#   C'est aussi la quantite d'action a detenir pour couvrir l'option (hedge).
def delta_call(S, K, T, r, sigma):
    return norm.cdf(_d1(S, K, T, r, sigma))

def delta_put(S, K, T, r, sigma):
    return norm.cdf(_d1(S, K, T, r, sigma)) - 1.0

# GAMMA — vitesse a laquelle le delta lui-meme change.
#   Identique pour call et put. Maximal a la monnaie (Day 2 : la ou la valeur
#   temps est la plus forte).
def gamma(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# VEGA — sensibilite a une variation de la volatilite (ici, pour +1 point de %,
#   d'ou la division par 100). Identique call/put. C'est le Greek qui relie
#   directement au concept de valeur temps.
def vega(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0

# THETA — decroissance temporelle : perte de valeur par jour qui passe.
#   On divise par 365 pour l'exprimer "par jour calendaire".
#   Generalement negatif pour un acheteur (Day 2 : la valeur temps s'erode).
def theta_call(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    return (term1 + term2) / 365.0

def theta_put(S, K, T, r, sigma):
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return (term1 + term2) / 365.0

# RHO — sensibilite a une variation du taux sans risque (pour +1 point de %,
#   d'ou /100). Le moins important en pratique, mais il fait partie des cinq.
def rho_call(S, K, T, r, sigma):
    d2 = _d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0

def rho_put(S, K, T, r, sigma):
    d2 = _d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


# ------------------------------------------------------------------------------
# 4. Verification : la parite put-call
# ------------------------------------------------------------------------------
# Une relation qui DOIT etre vraie : C - P = S - K*e^(-rT).
# Si elle tient, c'est un bon signe que le pricing est correct.
def check_put_call_parity(S, K, T, r, sigma):
    C = bs_call_price(S, K, T, r, sigma)
    P = bs_put_price(S, K, T, r, sigma)
    gauche = C - P
    droite = S - K * np.exp(-r * T)
    return gauche, droite, np.isclose(gauche, droite)


# ------------------------------------------------------------------------------
# 5. Demonstration
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Un cas simple : action a 100, strike 100 (a la monnaie), 1 an, r=5%, vol=20%
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    print("=" * 60)
    print("STAGE 1 — Black-Scholes pricer + Greeks")
    print("=" * 60)
    print(f"Entrees : S={S}, K={K}, T={T}an, r={r:.0%}, sigma={sigma:.0%}\n")

    call = bs_call_price(S, K, T, r, sigma)
    put = bs_put_price(S, K, T, r, sigma)
    print(f"Prix call : {call:7.4f}")
    print(f"Prix put  : {put:7.4f}\n")

    # Rappel Day 2 : ici l'option est A LA MONNAIE, donc valeur intrinseque = 0.
    # Tout le prix est de la VALEUR TEMPS.
    intrinseque_call = max(S - K, 0)
    print(f"Valeur intrinseque call : {intrinseque_call}")
    print(f"Valeur temps call       : {call - intrinseque_call:.4f}  <- tout le prix\n")

    print("Les cinq Greeks (call) :")
    print(f"  delta : {delta_call(S, K, T, r, sigma):+.4f}")
    print(f"  gamma : {gamma(S, K, T, r, sigma):+.4f}")
    print(f"  vega  : {vega(S, K, T, r, sigma):+.4f}   (par +1% de vol)")
    print(f"  theta : {theta_call(S, K, T, r, sigma):+.4f}   (par jour)")
    print(f"  rho   : {rho_call(S, K, T, r, sigma):+.4f}   (par +1% de taux)\n")

    g, d, ok = check_put_call_parity(S, K, T, r, sigma)
    print(f"Parite put-call : C-P={g:.4f}  vs  S-Ke^(-rT)={d:.4f}  -> {'OK' if ok else 'ERREUR'}")