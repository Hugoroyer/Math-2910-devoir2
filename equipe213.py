import math
from bissection import Bissection
from pointfixe import PointFixe

def F(Q):
    return math.exp(Q) + Q/2 - 5
res = Bissection(F, 1.0, 2.0, tol=5e-5, nmax=50)
print(res)

def g1(Q):
    return math.log(5 - Q/2)

def g2(Q):
    return 10 - 2*math.exp(Q)

def gN(Q):
    f = math.exp(Q) + Q/2 - 5
    fp = math.exp(Q) + 0.5
    return Q - f/fp

def print_table(iters, titre, max_rows=None):
    """
    iters: liste [Q0, Q1, ..., QN]
    max_rows: si None -> imprime tout; sinon imprime jusqu'à Q_max_rows
             ex: max_rows=5 -> imprime n=0..5
    """
    if max_rows is None:
        N = len(iters) - 1
    else:
        N = min(max_rows, len(iters) - 1)

    print("\n" + "="*80)
    print(titre)
    print("="*80)
    print(f"{'n':>3} {'Qn':>20} {'|en|≈|Qn-Qn-1|':>18} {'|en+1/en|':>12} {'|en+1/en^2|':>14} {'|en+1/en^3|':>14}")

    # Pré-calcul des "erreurs" approx: e_n ≈ |Qn - Qn-1|
    e = [None]*(N+1)
    e[0] = None
    for n in range(1, N+1):
        e[n] = abs(iters[n] - iters[n-1])

    for n in range(0, N+1):
        Qn = iters[n]

        # colonnes ratios (définies quand on a e[n] et e[n+1])
        r1 = r2 = r3 = None
        if n >= 1 and n+1 <= N and e[n] is not None and e[n] != 0:
            r1 = e[n+1] / e[n]
            r2 = e[n+1] / (e[n]**2) if e[n] != 0 else None
            r3 = e[n+1] / (e[n]**3) if e[n] != 0 else None

        # format affichage
        en_str = "-" if n == 0 else f"{e[n]:.10e}"
        r1_str = "-" if r1 is None else f"{r1:.6f}"
        r2_str = "-" if r2 is None else f"{r2:.6f}"
        r3_str = "-" if r3 is None else f"{r3:.6f}"

        print(f"{n:>3} {Qn:>20.15f} {en_str:>18} {r1_str:>12} {r2_str:>14} {r3_str:>14}")

def main():
    Q0 = 1.0
    tolr = 1e-8
    nmax = 150

    # g1 : on imprime toutes les itérations retournées (jusqu'à convergence)
    iters_g1 = [1.0] + PointFixe(g1, Q0, tolr, nmax)
    print_table(iters_g1, "TABLEAU 1 — Point fixe avec g1(Q)=ln(5-Q/2) (convergence attendue)")

    # g2 : on imprime seulement jusqu'à Q5 (n=0..5)
    iters_g2 = [1.0] + PointFixe(g2, Q0, tolr, nmax)
    print_table(iters_g2, "TABLEAU 2 — Point fixe avec g2(Q)=10-2e^Q (divergence attendue)", max_rows=5)

    iters_gN = PointFixe(gN, Q0, tolr, nmax)
    # si ton PointFixe NE retourne PAS Q0, décommente :
    iters_gN = [Q0] + iters_gN
    print_table(iters_gN, "TABLEAU 3 — Newton via gN(Q) (question L)")

if __name__ == "__main__":
    main()