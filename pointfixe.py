import time

def Bissection(fonc_pointfixe, Q0, tolr, nmax):
    # fonc_pointfixe is the function considered
    # Q0 is the initial value of the fixed point iteration
    # tolr is the tolerance for convergence
    # nmax is the maximum number of iterations


    X = [0] * nmax

    for i in range(nmax + 1):

        if i == 0:
            print(f'{i} Q0 = {Q0:E}')
            time.sleep(0.5)

        elif i == 1:
            Q1 = fonc_pointfixe(Q0)

            # erreur relative : |Q1 - Q0| / |Q1|
            if Q1 != 0:
                err_rel = abs(Q1 - Q0) / abs(Q1)
            else:
                err_rel = abs(Q1 - Q0)

            print(f'{i} Q0 = {Q0:E}, Q1 = {Q1:E}, err_rel = {err_rel:E}')
            time.sleep(0.5)

            X[i-1] = Q1

            if err_rel < tolr:
                print('La méthode du point fixe a convergé!')
                return [Q0] + X[:i]

            Q0 = Q1

        else:
            Qn = fonc_pointfixe(Q0)

            # erreur relative : |Qn - Qn-1| / |Qn|
            if Qn != 0:
                err_rel = abs(Qn - Q0) / abs(Qn)
            else:
                err_rel = abs(Qn - Q0)

            print(f'{i} Qn-1 = {Q0:E}, Qn = {Qn:E}, err_rel = {err_rel:E}')
            time.sleep(0.5)

            X[i-1] = Qn

            if err_rel < tolr:
                print('La méthode du point fixe a convergé!')
                return [Q0] + X[:i]

            Q0 = Qn

    print('La méthode du point fixe n\'a pas convergé...')
    return [Q0] + X