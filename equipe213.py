import math
from bissection import Bissection

def F(Q):
    return math.exp(Q) + 0.5*Q**2 - 5

res = Bissection(F, 1.0, 2.0, tol=5e-5, nmax=50)
print(res)