import math
from bissection import Bissection  # ou le nom exact de ton fichier

def F(Q):
    return math.exp(Q) + Q/2 - 5

X = Bissection(F, 1, 2, 5e-6, 25)  # devrait retourner une liste de ~18 valeurs

n = 17
Qn = X[n-1]          # Q17
Qn1 = X[n]           # Q18
err_vraie = abs(Qn1 - Qn)

print("Qn =", Qn)
print("|Qn+1 - Qn| =", err_vraie)
