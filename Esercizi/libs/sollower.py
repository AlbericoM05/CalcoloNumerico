import numpy as np
from numpy.linalg import norm

def sollower(L, b):
    """
    sollower - Soluzione di sistema triangolare inferiore non singolare (per righe)
    SYNOPSIS
      x = sollower(L, b)
    INPUT
      L (float array) - Matrice triangolare inferiore non singolare
      b (float array) - Vettore dei termini noti
    OUTPUT
      x (float array) - Vettore soluzione del sistema
    """
    [m, n] = L.shape; eps = np.finfo(np.float64).eps
    if ( m != n ):
        raise ValueError("Matrice dei coefficienti non quadrata")
    elif ( any( abs( np.diag(L) ) < eps * norm(L, np.inf) ) ):
        raise ValueError("Almeno un elemento diagonale di L e' numericamente troppo piccolo")
    x = b.copy()
    x[0] /= L[0, 0]
    for i in range(1, m):
        # prodotto scalare: operazione di tipo BLAS1
        x[i] -= np.dot( L[i, 0 : i], x[0 : i] )
        x[i] /= L[i, i]
    return x
# fine della funzione sollower

n = 4
L = np.tril( np.random.rand(n*n).reshape((n, n)) )
b = np.matmul( L, np.ones(n) )
x = sollower(L, b)
print(x)