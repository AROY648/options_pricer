"""
================================================================================
STAGE 2 — Visualisation : convergence binomial -> Black-Scholes
================================================================================
On trace le prix binomial du call en fonction du nombre de pas N, et on
superpose la ligne horizontale du prix Black-Scholes.

Ce qu'on voit :
  - pour N petit, le prix binomial OSCILLE autour de la valeur Black-Scholes
  - quand N grandit, les oscillations s'amortissent et le binomial se colle
    sur la ligne Black-Scholes.
C'est la preuve visuelle que les deux modeles decrivent la meme realite :
l'un en temps discret (arbre), l'autre en temps continu (formule).
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from black_scholes import bs_call_price
from binomial import binomial_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

# Prix Black-Scholes = la reference (une constante)
bs = bs_call_price(S, K, T, r, sigma)

# Prix binomial pour chaque N de 1 a 200
Ns = np.arange(1, 201)
prices = [binomial_price(S, K, T, r, sigma, int(N), "call", american=False)
          for N in Ns]

plt.figure(figsize=(11, 6))
plt.plot(Ns, prices, color="steelblue", linewidth=1,
         label="Prix binomial (CRR)")
plt.axhline(bs, color="crimson", linestyle="--", linewidth=2,
            label=f"Prix Black-Scholes = {bs:.4f}")
plt.title("Convergence du prix binomial vers Black-Scholes")
plt.xlabel("Nombre de pas N dans l'arbre")
plt.ylabel("Prix du call")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("stage2_convergence.png", dpi=130, bbox_inches="tight")
plt.show()
print("Graphe sauvegarde : stage2_convergence.png")