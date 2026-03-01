"""
equipeXX.py — Devoir 2 (H26) — MAT-2910 Analyse numérique pour l'ingénierie
Offre et demande : O(Q) = e^Q,  D(Q) = -Q^2 + 5
Racine de f(Q) = e^Q + Q^2 - 5 = 0 sur [1, 2]
"""

import math
import numpy as np
from bissection import Bissection
from pointfixe import Pointfixe


# Fonctions de base

def f(Q):
    """f(Q) = e^Q + Q^2 - 5  (offre - demande)"""
    return np.exp(Q) + Q**2 - 5

def O(Q):
    """Offre : O(Q) = e^Q"""
    return np.exp(Q)

def D(Q):
    """Demande : D(Q) = -Q^2 + 5"""
    return -Q**2 + 5

# Fonctions de point fixe

def g1(Q):
    """g1(Q) = ln(-Q^2 + 5)  — convergence linéaire"""
    return np.log(-Q**2 + 5)

def g2(Q):
    """g2(Q) = 10 - 2*e^Q  — divergence"""
    return 10 - 2 * np.exp(Q)

def gN(Q):
    """Newton : gN(Q) = Q - f(Q)/f'(Q),  f'(Q) = e^Q + 2Q"""
    return Q - f(Q) / (np.exp(Q) + 2 * Q)

def steffenson(g):
    """
    Transforme g en la fonction de Steffenson :
      g_Steff(Q) = Q - (g(Q) - Q)^2 / (g(g(Q)) - 2*g(Q) + Q)
    """
    def g_steff(Q):
        gQ  = g(Q)
        ggQ = g(gQ)
        denom = ggQ - 2 * gQ + Q
        if abs(denom) < 1e-15:
            raise ZeroDivisionError("Dénominateur nul dans Steffenson")
        return Q - (gQ - Q)**2 / denom
    return g_steff


# ─────────────────────────────────────────────────────────────────────────────
# Affichage de tableau
# ─────────────────────────────────────────────────────────────────────────────

def afficher_tableau(titre, iterations, max_iter=None):
    """
    Affiche un tableau n | Qn | |en| | |Qn-Qn-1| | ratios d'ordre 1,2,3.
    Q_ref = dernière valeur de la liste utilisée comme meilleure approximation.
    """
    iters = list(iterations[:max_iter]) if max_iter is not None else list(iterations)
    Q_ref = iters[-1]
    errors = [abs(Q - Q_ref) for Q in iters]

    sep = "-" * 102
    print(f"\n{sep}")
    print(f"  {titre}")
    print(sep)
    print(f"{'n':>4}  {'Qn':>22}  {'|en|':>15}  {'|Qn-Qn-1|':>13}  "
          f"{'|en+1/en|':>12}  {'|en+1/en^2|':>13}  {'|en+1/en^3|':>13}")
    print(sep)

    for i, Q in enumerate(iters):
        e_c = errors[i]
        e_n = errors[i + 1] if i + 1 < len(iters) else None

        diff = abs(Q - iters[i - 1]) if i > 0 else None
        r1 = abs(e_n / e_c)      if (e_n is not None and e_c > 1e-30) else None
        r2 = abs(e_n / e_c**2)   if (e_n is not None and e_c > 1e-30) else None
        r3 = abs(e_n / e_c**3)   if (e_n is not None and e_c > 1e-30) else None

        s_e    = f"{e_c:.9e}"   if i < len(iters) - 1 else "—"
        s_diff = f"{diff:.5e}"  if diff is not None    else "—"
        s_r1   = f"{r1:.6f}"   if r1   is not None    else "—"
        s_r2   = f"{r2:.6f}"   if r2   is not None    else "—"
        s_r3   = f"{r3:.6f}"   if r3   is not None    else "—"

        print(f"{i:>4}  {Q:>22.16f}  {s_e:>15}  {s_diff:>13}  "
              f"{s_r1:>12}  {s_r2:>13}  {s_r3:>13}")

    print(sep)


# Question B — Nombre d'itérations de bissection

print("=" * 102)
print("PARTIE B — Nombre d'itérations pour 5 chiffres significatifs")
print("=" * 102)

# Qexacte ~ 1.xxxx  ->  5 chiffres sig.  ->  erreur absolue < 5e-5
# Condition : (b-a)/2^n = 1/2^n < 5e-5
# => n > log2(20000) ≈ 14.29  =>  n_min = 15

n_biss   = 15
tol_biss = 5e-5

print(f"  Tolérance cible  : {tol_biss:.1e}  (demi-unité du 4e chiffre décimal pour Qexacte ~ 1.xxxx)")
print(f"  (b-a)/2^n = 1/2^n < {tol_biss:.1e}  =>  n > log2({1/tol_biss:.0f}) = {math.log2(1/tol_biss):.4f}")
print(f"  =>  n_min = {n_biss} itérations")


# Question C — Bissection numérique
#
# Bissection(F, x1, x2, tol, nmax) retourne X[:i] = [Q1, Q2, ..., Q_i]
# où Q_k = X[k-1] est le point milieu de la k-ième itération.
# On lance avec nmax=16 et tol=0 pour obtenir Q1…Q16 sans arrêt anticipé.

print("\n" + "=" * 102)
print("PARTIE C — Bissection numérique (nmax=16, tol=0)")
print("=" * 102)

print("\n--- Appel Bissection ---")
X_biss = Bissection(f, 1.0, 2.0, 0.0, 16)

Qn   = X_biss[n_biss - 1]   # Q_15  (index 14)
Qn1  = X_biss[n_biss]       # Q_16  (index 15)
err_vraie = abs(Qn1 - Qn)

print(f"\n  Q_n   = Q_{n_biss}  = {Qn:.10f}")
print(f"  Q_n+1 = Q_{n_biss+1} = {Qn1:.10f}")
print(f"  |Q_n+1 - Q_n| = {err_vraie:.2e}  ", end="")
print("< tol_biss  ✓" if err_vraie < tol_biss else ">= tol_biss  ✗")


# Question D — Propagation d'erreur

print("\n" + "=" * 102)
print("PARTIE D — Propagation d'erreur : δO et δD")
print("=" * 102)

dO_val = np.exp(Qn)      # |O'(Q)| = e^Q
dD_val = abs(-2 * Qn)    # |D'(Q)| = 2|Q|

delta_O = dO_val * err_vraie
delta_D = dD_val * err_vraie

print(f"  Qn = {Qn:.8f},   |erreur| <= {err_vraie:.2e}")
print(f"  |O'(Qn)| = e^Qn     = {dO_val:.6f}")
print(f"  |D'(Qn)| = 2*Qn     = {dD_val:.6f}")
print(f"  delta_O <= |O'(Qn)| * erreur = {delta_O:.2e}")
print(f"  delta_D <= |D'(Qn)| * erreur = {delta_D:.2e}")
print()
print("  Explication : bien que O(Qexacte) = D(Qexacte), les bornes d'erreur different")
print("  car elles font intervenir les DERIVEES : |O'| = e^Q != |D'| = 2Q.")


# Question E — Chiffres significatifs de O(Qn) et D(Qn)

print("\n" + "=" * 102)
print("PARTIE E — O(Qn) et D(Qn) avec leurs chiffres significatifs")
print("=" * 102)

OQn = O(Qn)
DQn = D(Qn)

def nb_chiffres_sig(val, delta):
    if delta == 0 or val == 0:
        return float('inf')
    return max(0, math.floor(math.log10(abs(val))) - math.floor(math.log10(delta)) + 1)

def arrondi_sig(val, n_sig):
    if val == 0 or n_sig <= 0:
        return 0.0
    d = -int(math.floor(math.log10(abs(val)))) + (n_sig - 1)
    return round(val, d)

ns_O = nb_chiffres_sig(OQn, delta_O)
ns_D = nb_chiffres_sig(DQn, delta_D)

print(f"  O(Qn) = {OQn:.10f}   delta_O = {delta_O:.2e}   -> {ns_O} chiffres significatifs fiables")
print(f"  D(Qn) = {DQn:.10f}   delta_D = {delta_D:.2e}   -> {ns_D} chiffres significatifs fiables")
print(f"\n  O(Qn) arrondi : {arrondi_sig(OQn, ns_O)}")
print(f"  D(Qn) arrondi : {arrondi_sig(DQn, ns_D)}")


# Question J — Point fixe : g1 (convergence) et g2 (divergence)

print("\n" + "=" * 102)
print("PARTIE J — Point fixe : g1 et g2")
print("=" * 102)

Q0   = 1.0
tolr = 1e-8
nmax = 150

print("\n--- Pointfixe(g1) ---")
iters_g1 = Pointfixe(g1, Q0, tolr, nmax)
afficher_tableau("Tableau 1 — g1(Q) = ln(-Q^2+5)  [convergence d'ordre 1]", iters_g1)

print("\n--- Pointfixe(g2) — 5 premieres iterations ---")
iters_g2 = Pointfixe(g2, Q0, tolr, nmax)
afficher_tableau("Tableau 2 — g2(Q) = 10 - 2*e^Q  [divergence]", iters_g2, max_iter=6)


# Question L — Newton (gN comme point fixe)

print("\n" + "=" * 102)
print("PARTIE L — Newton via gN(Q) = Q - f(Q)/f'(Q)")
print("=" * 102)

iters_gN = Pointfixe(gN, Q0, tolr, nmax)
afficher_tableau("Tableau 3 — Newton gN(Q)  [convergence d'ordre 2]", iters_gN)
print("\n  Commentaire : le ratio |e_{n+1}/e_n^2| se stabilise vers une constante,")
print("  confirmant la convergence quadratique (ordre 2) attendue de Newton.")

# Question M — Steffenson sur g1, g2, gN

print("\n" + "=" * 102)
print("PARTIE M — Methode de Steffenson")
print("=" * 102)

g_s1 = steffenson(g1)
g_s2 = steffenson(g2)
g_sN = steffenson(gN)

print("\n--- Steffenson(g1) ---")
iters_s1 = Pointfixe(g_s1, Q0, tolr, nmax)
afficher_tableau("Tableau 4 — Steffenson(g1)  [g1 lineaire -> ordre 2]", iters_s1)

print("\n--- Steffenson(g2) ---")
iters_s2 = Pointfixe(g_s2, Q0, tolr, nmax)
afficher_tableau("Tableau 5 — Steffenson(g2)  [g2 divergeait -> Steffenson corrige]", iters_s2)

print("\n--- Steffenson(gN) ---")
iters_sN = Pointfixe(g_sN, Q0, tolr, nmax)
afficher_tableau("Tableau 6 — Steffenson(gN)  [gN ordre 2 -> ordre 4 attendu]", iters_sN)

print("\n  Commentaire : Steffenson transforme une methode d'ordre p en ordre 2p.")
print("  - Steff(g1) : ordre 1 -> 2 (ratio |e_{n+1}/e_n^2| constant)")
print("  - Steff(g2) : divergence -> convergence (Steffenson 'repare' g2)")
print("  - Steff(gN) : ordre 2 -> 4 (ratio |e_{n+1}/e_n^4| constant)")

print("\n" + "=" * 102)
print("Fin du script equipe213.py")
print("=" * 102)