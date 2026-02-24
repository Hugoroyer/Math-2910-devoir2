import time

def Bissection(F, x1, x2, tol, nmax):
    # F=F(x) is the function considered
    # x1 is the lower bound of the interval considered
    # x2 is the upper bound of the interval considered
    # tol is the tolerance
    # nmax is the maximum number of iterations

    F1 = F(x1)
    F2 = F(x2)

    if F1 * F2 > 0:
        print('La méthode de la bissection n\'a pas convergé...')
        print('L\'intervalle de départ est incorrect.')
        return x1

    if x2 < x1:
        x1, x2 = x2, x1

    X = [0] * nmax

    for i in range(nmax + 1):

        if i == 0:
            print(f'{i} x1 = {x1:E}, x2 = {x2:E}, F(x1) = {F1:E}, F(x2) = {F2:E}')
            time.sleep(0.5)

            if F1 == 0:
                print('La méthode de la bissection a convergé!')
                return x1
            elif F2 == 0:
                print('La méthode de la bissection a convergé!')
                return x2

        elif i == 1:
            xmilieu = (x1 + x2) / 2
            Fmilieu = F(xmilieu)

            longueurIntervalle = (x2 - x1) / 2

            print(f'{i} x1 = {x1:E}, x2 = {x2:E}, xm = {xmilieu:E}, F(x1) = {F1:E}, F(x2) = {F2:E}, F(xm) = {Fmilieu:E}, err = {longueurIntervalle:E}')
            time.sleep(0.5)

            X[i-1] = xmilieu

            if longueurIntervalle < tol or Fmilieu == 0:
                print('La méthode de la bissection a convergé!')
                return X[:i]

            if F1 * Fmilieu > 0:
                x1 = xmilieu
                F1 = Fmilieu
            else:
                x2 = xmilieu
                F2 = Fmilieu

            longueurDepart = 2 * longueurIntervalle

        else:
            xmilieu = (x1 + x2) / 2
            Fmilieu = F(xmilieu)

            longueurIntervalle /= 2

            ratio1 = 0.5
            ratio2 = 2 ** (i - 1) / longueurDepart

            print(f'{i} x1 = {x1:E}, x2 = {x2:E}, xm = {xmilieu:E}, F(x1) = {F1:E}, F(x2) = {F2:E}, F(xm) = {Fmilieu:E}, err = {longueurIntervalle:E}, Ratio1 = {ratio1:E}, Ratio 2 = {ratio2:E}')
            time.sleep(0.5)

            X[i-1] = xmilieu

            if longueurIntervalle < tol or Fmilieu == 0:
                print('La méthode de la bissection a convergé!')
                return X[:i]

            if F1 * Fmilieu > 0:
                x1 = xmilieu
                F1 = Fmilieu
            else:
                x2 = xmilieu
                F2 = Fmilieu

    print('La méthode de la bissection n\'a pas convergé...')

    

