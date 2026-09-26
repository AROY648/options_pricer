"""
Visualisation des greeks — reutilise les fonctions de pricer.py.
Genere deux figures :
  1) greeks_vs_spot.png     : les 6 greeks en fonction du spot (a T fixe)
  2) greeks_dynamics.png    : comment gamma et vega evoluent avec l'echeance
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pricer as p

K, r = 7750, 0.0462
plt.rcParams.update({"figure.dpi":130, "font.size":10, "axes.grid":True,
                     "grid.alpha":0.25, "axes.spines.top":False, "axes.spines.right":False})

def spot_axis(K, span=0.10, n=300):
    return np.linspace(K*(1-span), K*(1+span), n)

# ---------------------------------------------------------------- Figure 1
def fig_greeks_vs_spot(T=38/365, sigma=0.1264):
    S = spot_axis(K)
    panels = [
        ("Delta",  p.delta_call(S,K,T,r,sigma), "sensibilite au spot (courbe en S)"),
        ("Gamma",  p.gamma(S,K,T,r,sigma),      "vitesse du delta — cloche ATM"),
        ("Theta / jour", p.theta_call(S,K,T,r,sigma), "decroissance — creux ATM"),
        ("Vega (+1%)", p.vega(S,K,T,r,sigma),   "sensibilite a la vol — cloche ATM"),
        ("Vanna", p.vanna(S,K,T,r,sigma),       "d(delta)/d(vol) — nulle ATM, ailes"),
        ("Volga", p.volga(S,K,T,r,sigma),       "convexite du vega — pics dans les AILES"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax,(name,y,sub) in zip(axes.ravel(), panels):
        ax.plot(S, y, color="#2b6cf6", lw=2)
        ax.axvline(K, ls="--", color="#999", lw=1)
        if y.min()<0<y.max(): ax.axhline(0, color="#ccc", lw=1)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel(sub, fontsize=8, color="#555")
        ax.tick_params(labelsize=8)
    fig.suptitle(f"Les greeks vs spot   (K={K}, T={T*365:.0f}j, IV={sigma:.1%})   — ligne pointillee = ATM",
                 fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig("greeks_vs_spot.png", bbox_inches="tight")

# ---------------------------------------------------------------- Figure 2
def fig_dynamics(sigma=0.1264):
    S = spot_axis(K, span=0.08)
    fig, (axg, axv) = plt.subplots(1, 2, figsize=(11, 4.4))

    # gamma qui explose quand l'echeance approche
    for days,c in [(60,"#9ec5ff"),(20,"#4b8bff"),(5,"#0a3fa8")]:
        axg.plot(S, p.gamma(S,K,days/365,r,sigma), color=c, lw=2, label=f"{days} j")
    axg.axvline(K, ls="--", color="#999", lw=1)
    axg.set_title("GAMMA explose a l'ATM pres de l'echeance", fontweight="bold")
    axg.legend(title="jours restants"); axg.set_xlabel("spot")

    # vega qui grandit avec l'echeance
    for days,c in [(10,"#ffc19e"),(40,"#ff7a4b"),(90,"#c0330a")]:
        axv.plot(S, p.vega(S,K,days/365,r,sigma), color=c, lw=2, label=f"{days} j")
    axv.axvline(K, ls="--", color="#999", lw=1)
    axv.set_title("VEGA grandit avec l'echeance", fontweight="bold")
    axv.legend(title="jours restants"); axv.set_xlabel("spot")

    fig.suptitle("Les greeks sont DYNAMIQUES — memes formules, echeances differentes",
                 fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig("greeks_dynamics.png", bbox_inches="tight")

if __name__ == "__main__":
    fig_greeks_vs_spot()
    fig_dynamics()
    print("Figures generees : greeks_vs_spot.png, greeks_dynamics.png")