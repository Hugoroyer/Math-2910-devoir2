import sys
import io
import numpy as np
from bissection import Bissection
from pointfixe import Pointfixe


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions de base
# ─────────────────────────────────────────────────────────────────────────────

def f(Q):
    """f(Q) = e^Q + Q^2 - 5"""
    return np.exp(Q) + Q**2 - 5

def g1(Q):
    """g1(Q) = ln(-Q^2 + 5)"""
    return np.log(-Q**2 + 5)

def g2(Q):
    """g2(Q) = 10 - 2*e^Q"""
    return 10 - 2 * np.exp(Q)

def gN(Q):
    """Newton : gN(Q) = Q - f(Q)/f'(Q),  f'(Q) = e^Q + 2Q"""
    return Q - f(Q) / (np.exp(Q) + 2 * Q)

def steffenson(g):
    """Retourne la fonction de point fixe de Steffenson basée sur g."""
    def g_steff(Q):
        gQ  = g(Q)
        ggQ = g(gQ)
        denom = ggQ - 2 * gQ + Q
        if abs(denom) < 1e-15:
            raise ZeroDivisionError("Denominateur nul dans Steffenson")
        return Q - (gQ - Q)**2 / denom
    return g_steff


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaire d'affichage de tableau
# Modèle : n | Qn | |en|~|Qn-Qn-1| | |en+1/en| | |en+1/en²| | |en+1/en³|
# ─────────────────────────────────────────────────────────────────────────────

def afficher_tableau(titre, iterations, max_iter=None):
    iters = list(iterations[:max_iter]) if max_iter is not None else list(iterations)

    # Différences consécutives comme approximation de l'erreur
    diffs = [None] + [abs(iters[i] - iters[i-1]) for i in range(1, len(iters))]

    sep = "-" * 100
    print(f"\n{sep}")
    print(f"  {titre}")
    print(sep)
    print(f"{'n':>4}  {'Qn':>22}  {'|en|~|Qn-Qn-1|':>16}  "
          f"{'|en+1/en|':>12}  {'|en+1/en^2|':>13}  {'|en+1/en^3|':>13}")
    print(sep)

    for i, Q in enumerate(iters):
        e_c = diffs[i]
        e_n = diffs[i + 1] if i + 1 < len(iters) else None

        r1 = abs(e_n / e_c)    if (e_c and e_n and e_c > 1e-30) else None
        r2 = abs(e_n / e_c**2) if (e_c and e_n and e_c > 1e-30) else None
        r3 = abs(e_n / e_c**3) if (e_c and e_n and e_c > 1e-30) else None

        s_e  = f"{e_c:.9e}" if e_c is not None else "→"
        s_r1 = f"{r1:.6f}" if r1  is not None else "→"
        s_r2 = f"{r2:.6f}" if r2  is not None else "→"
        s_r3 = f"{r3:.6f}" if r3  is not None else "→"

        print(f"{i:>4}  {Q:>22.16f}  {s_e:>16}  "
              f"{s_r1:>12}  {s_r2:>13}  {s_r3:>13}")

    print(sep)


def run_silent(func, *args, **kwargs):
    """Exécute func sans afficher sa sortie console."""
    _out = sys.stdout
    sys.stdout = io.StringIO()
    result = func(*args, **kwargs)
    sys.stdout = _out
    return result


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE C — Bissection numérique (n = 15 itérations)
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 100)
print("PARTIE C — Bissection numerique")
print("=" * 100)

n_biss = 15

# tol=1e-15 et nmax=60 pour obtenir suffisamment d'itérations sans retour None
X_biss = Bissection(f, 1.0, 2.0, 1e-15, 60)

Qn       = X_biss[n_biss - 1]   # Q_15
Qn1      = X_biss[n_biss]       # Q_16
err_vraie = abs(Qn1 - Qn)

print(f"\n  Q_n   = Q_{n_biss}  = {Qn:.10f}")
print(f"  Q_n+1 = Q_{n_biss+1} = {Qn1:.10f}")
print(f"  |Q_n+1 - Q_n| = {err_vraie:.2e}  "
      + ("< 5e-5  (5 chiffres significatifs confirmes)" if err_vraie < 5e-5 else ">= 5e-5"))


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE J — Point fixe : g1 (convergence) et g2 (divergence)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("PARTIE J — Methode de point fixe : g1 et g2")
print("=" * 100)

Q0   = 1.0
tolr = 1e-8
nmax = 150

iters_g1 = run_silent(Pointfixe, g1, Q0, tolr, nmax)
afficher_tableau("Tableau 1 — g1(Q) = ln(-Q^2+5)  [convergence d'ordre 1]",
                 iters_g1, max_iter=6)

iters_g2 = run_silent(Pointfixe, g2, Q0, tolr, nmax)
afficher_tableau("Tableau 2 — g2(Q) = 10 - 2*e^Q  [divergence — 5 premieres iterations]",
                 iters_g2, max_iter=6)


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE L — Methode de Newton via gN comme point fixe
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("PARTIE L — Methode de Newton  gN(Q) = Q - f(Q)/f'(Q)")
print("=" * 100)

iters_gN = run_silent(Pointfixe, gN, Q0, tolr, nmax)
afficher_tableau("Tableau 3 — Newton gN(Q)  [convergence d'ordre 2]", iters_gN)


# ═════════════════════════════════════════════════════════════════════════════
# PARTIE M — Methode de Steffenson sur g1, g2, gN
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("PARTIE M — Methode de Steffenson")
print("=" * 100)

iters_s1 = run_silent(Pointfixe, steffenson(g1), Q0, tolr, nmax)
afficher_tableau("Tableau 4 — Steffenson(g1)  [ordre 1 -> ordre 2]", iters_s1)

iters_s2 = run_silent(Pointfixe, steffenson(g2), Q0, tolr, nmax)
afficher_tableau("Tableau 5 — Steffenson(g2)  [divergeait -> converge]", iters_s2)

iters_sN = run_silent(Pointfixe, steffenson(gN), Q0, tolr, nmax)
afficher_tableau("Tableau 6 — Steffenson(gN)  [ordre 2 -> ordre 4]", iters_sN)

print("\n" + "=" * 100)
print("Fin du script equipe213.py")
print("=" * 100)