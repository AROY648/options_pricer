"""
================================================================================
STAGE 2 — Arbre binomial (Cox-Ross-Rubinstein)
================================================================================

Deuxieme methode de pricing, en complement de Black-Scholes (Stage 1).

Ce que ce module apporte :
  1. Prix d'une option par arbre binomial (europeenne ET americaine).
  2. La preuve que le binomial CONVERGE vers Black-Scholes quand N grandit.
  3. Le pricing des options AMERICAINES, que Black-Scholes ne sait pas faire.

--------------------------------------------------------------------------------
L'IDEE (rappel Stage 2)
--------------------------------------------------------------------------------
On decoupe le temps T en N petits pas de duree dt = T/N.
A chaque pas, l'action peut MONTER (x u) ou DESCENDRE (x d).

Modele CRR :
    u = e^( sigma * sqrt(dt) )      (facteur de hausse)
    d = 1 / u                       (facteur de baisse)
    p = ( e^(r*dt) - d ) / (u - d)  (proba risque-neutre de hausse)

Le "sigma*sqrt(dt)" est le meme "quantite de mouvement" que le sigma*sqrt(T)
du Day 3, mais decoupe en petits pas.

On price par BACKWARD INDUCTION :
    - a l'echeance, on connait les payoffs (Day 1 : max(S-K,0) ou max(K-S,0))
    - on "remonte" l'arbre : la valeur a chaque noeud = esperance actualisee
      des deux noeuds enfants, sous la proba risque-neutre p.
================================================================================
"""

import numpy as np


def binomial_price(S, K, T, r, sigma, N, option_type="call", american=False):
    """
    Prix d'une option par arbre binomial CRR.

    Parametres
    ----------
    S, K, T, r, sigma : memes entrees que Black-Scholes (Stage 1)
    N                 : nombre de pas dans l'arbre (plus grand = plus precis)
    option_type       : "call" ou "put"
    american          : False = europeenne, True = americaine
                        (americaine = exercable a tout moment)
    """
    dt = T / N                          # duree d'un pas
    u = np.exp(sigma * np.sqrt(dt))     # facteur de hausse
    d = 1.0 / u                         # facteur de baisse
    p = (np.exp(r * dt) - d) / (u - d)  # proba risque-neutre
    disc = np.exp(-r * dt)              # actualisation d'un pas

    # --- 1. Prix du sous-jacent a l'echeance (les N+1 noeuds finaux) ---------
    # Au noeud final j : l'action a monte j fois et descendu (N-j) fois.
    j = np.arange(N + 1)
    ST = S * (u ** j) * (d ** (N - j))

    # --- 2. Payoff a l'echeance (Day 1) -------------------------------------
    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    # --- 3. Backward induction : on remonte l'arbre -------------------------
    for i in range(N - 1, -1, -1):
        # valeur "de continuation" = esperance actualisee des deux enfants
        values = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])

        if american:
            # option americaine : a chaque noeud, comparer
            # "exercer maintenant" vs "continuer". On garde le max.
            ST_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            if option_type == "call":
                exercise = np.maximum(ST_i - K, 0.0)
            else:
                exercise = np.maximum(K - ST_i, 0.0)
            values = np.maximum(values, exercise)

    return values[0]


# ------------------------------------------------------------------------------
# Demonstration
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from black_scholes import bs_call_price, bs_put_price

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    print("=" * 64)
    print("STAGE 2 — Arbre binomial (CRR)")
    print("=" * 64)
    print(f"Entrees : S={S}, K={K}, T={T}an, r={r:.0%}, sigma={sigma:.0%}\n")

    # --- Convergence vers Black-Scholes -------------------------------------
    bs_call = bs_call_price(S, K, T, r, sigma)
    print(f"Prix call Black-Scholes (reference) : {bs_call:.4f}\n")
    print("Prix binomial selon le nombre de pas N :")
    for N in [5, 10, 50, 100, 500, 1000]:
        bino = binomial_price(S, K, T, r, sigma, N, "call", american=False)
        ecart = bino - bs_call
        print(f"  N={N:>4} : {bino:.4f}   (ecart vs BS : {ecart:+.4f})")

    print("\n-> le prix binomial converge vers Black-Scholes quand N grandit.\n")

    # --- Option americaine : ce que Black-Scholes ne sait pas faire ---------
    N = 1000
    euro_put = binomial_price(S, K, T, r, sigma, N, "put", american=False)
    amer_put = binomial_price(S, K, T, r, sigma, N, "put", american=True)
    bs_put   = bs_put_price(S, K, T, r, sigma)

    print("Put europeen vs americain (N=1000) :")
    print(f"  Put europeen (binomial) : {euro_put:.4f}")
    print(f"  Put europeen (BS, ref)  : {bs_put:.4f}   <- concorde avec le binomial")
    print(f"  Put AMERICAIN (binomial): {amer_put:.4f}   <- BS ne peut PAS le calculer")
    print(f"  Prime d'exercice anticipe : {amer_put - euro_put:+.4f}")
    print("\n  Le put americain vaut PLUS : le droit d'exercer avant l'echeance")
    print("  a une valeur. C'est l'avantage de l'arbre sur Black-Scholes.")