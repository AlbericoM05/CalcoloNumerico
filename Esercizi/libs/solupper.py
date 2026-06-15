import numpy as np
from numpy.linalg import norm

def solupper(R, b):
    """
    solupper - Soluzione di sistema triangolare superiore non singolare (per righe)
    SYNOPSIS
      x = solupper(R, b)
    INPUT
      R (float array) - Matrice triangolare superiore non singolare
      b (float array) - Vettore dei termini noti
    OUTPUT
      x (float array) - Vettore soluzione del sistema
    """
    [m, n] = R.shape; eps = np.finfo(np.float64).eps
    if ( m != n ):
        raise ValueError("Matrice dei coefficienti non quadrata")
    elif ( any( abs( np.diag(R) ) < eps * norm(R, np.inf) ) ):
        raise ValueError("Almeno un elemento diagonale di R e' numericamente troppo piccolo")
    x = b.copy(); n -= 1
    x[n] /= R[n, n]
    for i in range( n-1, -1, -1 ):
        # prodotto scalare: operazione di tipo BLAS1
        x[i] -= np.dot( R[i, i+1 : ], x[ i+1 : ] )
        x[i] /= R[i, i]
    return x
# fine della funzione solupper

n = 4
R = np.triu( np.random.rand(n*n).reshape((n, n)) )
b = np.matmul( R, np.ones(n) )
x = solupper(R, b)
print(x)
