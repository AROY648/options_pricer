"""
================================================================================
STAGE 1 — Visualisations
================================================================================
Trois graphes pour VOIR ce que les Days 2-4 decrivent :

  (A) Prix du call vs prix du sous-jacent, compare au payoff a l'echeance.
      -> on voit la VALEUR TEMPS (Day 2) comme l'ecart entre la courbe lisse
         (avant echeance) et la ligne cassee (a l'echeance).

  (B) Delta du call vs prix du sous-jacent.
      -> on voit delta passer de 0 (OTM) a 1 (ITM), en S autour du strike.

  (C) Gamma et Theta vs prix du sous-jacent.
      -> gamma est maximal A LA MONNAIE (Day 2 : la ou la valeur temps est
         la plus forte) ; theta y est le plus negatif (decroissance maximale).
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from black_scholes import (
    bs_call_price, delta_call, gamma, theta_call
)

# Parametres fixes
K, T, r, sigma = 100.0, 1.0, 0.05, 0.20

# Une gamme de prix du sous-jacent autour du strike
S = np.linspace(50, 150, 300)

# Calculs vectorises (numpy gere le tableau S d'un coup)
prix = bs_call_price(S, K, T, r, sigma)
payoff = np.maximum(S - K, 0)          # valeur intrinseque = payoff a l'echeance
d = delta_call(S, K, T, r, sigma)
g = gamma(S, K, T, r, sigma)
th = theta_call(S, K, T, r, sigma)

# --- Figure a 3 panneaux -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (A) Prix vs payoff -> valeur temps
axes[0].plot(S, prix, label="Prix du call (avant echeance)", linewidth=2)
axes[0].plot(S, payoff, "--", label="Payoff a l'echeance (intrinseque)", linewidth=2)
axes[0].axvline(K, color="gray", linestyle=":", alpha=0.7, label="Strike K=100")
axes[0].fill_between(S, payoff, prix, alpha=0.15, label="Valeur temps")
axes[0].set_title("(A) Prix vs payoff : la valeur temps")
axes[0].set_xlabel("Prix du sous-jacent S")
axes[0].set_ylabel("Valeur de l'option")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# (B) Delta
axes[1].plot(S, d, color="darkorange", linewidth=2)
axes[1].axvline(K, color="gray", linestyle=":", alpha=0.7)
axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.4)
axes[1].set_title("(B) Delta du call : de 0 (OTM) a 1 (ITM)")
axes[1].set_xlabel("Prix du sous-jacent S")
axes[1].set_ylabel("Delta")
axes[1].grid(alpha=0.3)

# (C) Gamma et Theta
ax_g = axes[2]
ax_t = ax_g.twinx()   # deuxieme axe y pour theta (echelle differente)
l1, = ax_g.plot(S, g, color="seagreen", linewidth=2, label="Gamma")
l2, = ax_t.plot(S, th, color="crimson", linewidth=2, label="Theta (par jour)")
ax_g.axvline(K, color="gray", linestyle=":", alpha=0.7)
ax_g.set_title("(C) Gamma (max ATM) et Theta (min ATM)")
ax_g.set_xlabel("Prix du sous-jacent S")
ax_g.set_ylabel("Gamma", color="seagreen")
ax_t.set_ylabel("Theta", color="crimson")
ax_g.legend(handles=[l1, l2], fontsize=8, loc="upper right")
ax_g.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("stage1_greeks.png", dpi=130, bbox_inches="tight")
print("Graphe sauvegarde : stage1_greeks.png")