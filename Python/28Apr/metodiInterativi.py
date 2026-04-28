import numpy as np
from numpy.linalg import norm

def jacobi(A, b, x, maxit, tol):
    
    # Metodo iterativo di Jacobi per i sistemi lineari.

    # TODO: Controlli dell'input (es. matrice quadrata?).

    n = len(A)
    d = np.diag(A)                  # np.diag(A) restituisce la matrice diagonale di A.

    b = b / d                       # Equivale a D^-1 * b

    k = 0
    stop = False

    while not stop:
        k += 1
        xtemp = np.copy(x)                  # x al passo k
        x = x - (A @ x) / d + b     # METODO DI JACOBI
                                    # @ è la somma tra matrici e vettori
        stop = (norm(xtemp - x, np.inf) < tol * norm(x, np.inf)) or (k >= maxit)

    if(k == maxit):
        print(f"Raggiunto numero massimo di iterazioni: ", (maxit))

    return x, k

def gaussSeidel(A, b, x, maxit, tol):

    # Metodo iterativo di Gauss Seidel per sistemi lineari.

    # TODO: Controlli su input.

    n = len(A)

    k = 0
    stop = False

    while not stop:
        k += 1
        xtemp = np.copy(x)

        for i in range(n):
            x[i] = (b[i] - np.hstack((A[i, :i], A[i, i+1:])) @ np.hstack((x[:i], x[i+1:]))) / A[i, i]
        
        stop = (norm(xtemp - x, np.inf) < tol * norm(x, np.inf)) or (k >= maxit)

    if(k == maxit):
        print(f"Raggiunto numero massimo di iterazioni: ", (maxit))
    
    return x, k

def sor(A, b, x, maxit, tol, omega):

    # Metodo iterativo SOR per sistemi lineari.

    # TODO: Controlli su input.

    n = len(A)

    k = 0
    stop = False

    while not stop:
        k += 1
        xtemp = np.copy(x)

        for i in range(n):
            x[i] = (b[i] - np.hstack((A[i, :i], A[i, i+1:])) @ np.hstack((x[:i], x[i+1:]))) / A[i, i]
            x[i] = (1 - omega) * xtemp[i] + omega * x[i]
        
        stop = (norm(xtemp - x, np.inf) < tol * norm(x, np.inf)) or (k >= maxit)

    if(k == maxit):
        print(f"Raggiunto numero massimo di iterazioni: ", (maxit))
    
    return x, k