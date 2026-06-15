#------------------------------------------------------------------------------
#  Codici allegati alla prova scritta del 11/07/2018 - Versione Python
#------------------------------------------------------------------------------
import numpy as np
from numpy.linalg import norm 

#------------------------------------------------------------------------------
#  Schema di Ruffini-Horner
#------------------------------------------------------------------------------
def ruffiniHorner(p, a):
    """
    ruffiniHorner - Schema di Ruffini-Horner
      Calcola il valore di un polinomio p(x) nel punto x = a e i coefficienti
      del polinomio quoziente q(x) = p(x) / (x - a) 
    SYNOPSIS
      r, q = ruffiniHorner(p, a)
    INPUT
      p (float array)   - Vettore dei coefficienti del polinomio, ordinati
                          da quello di grado piu' alto a quello di grado zero
      a (float)         - Punto (numero reale) in cui calcolare il polinomio
    OUTPUT
      r (float64)       - Valore del polinomio nel punto x = a
      q (float64 array) - Vettore dei coefficienti del polinomio quoziente 
                          q(x) = p(x) / (x - a)
    """
    if ( (p is None) or (not p.size) ):   # vettore dei coefficienti vuoto 
        print("WARNING: il vettore p dei coefficienti e' vuoto")
        return [], []
    elif ( a is None ):  # punto non fornito: q = p (inalterato)
        print("WARNING: il punto 'a' in cui valutare il polinomio e' vuoto")
        return [], p
    else:
        n = p.size    # n - 1 = grado del polinomio
        q = np.zeros(n, dtype = np.float64)
        q[0] = p[0]
        for k in range(1, n):
            q[k] = q[k-1]*a + p[k]
        return q[-1], q[0:(n-1)]
# fine della funzione ruffiniHorner

#------------------------------------------------------------------------------
#  Algoritmo di sostituzione in avanti per sistemi triangolari inferiori
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
#  Algoritmo di sostituzione in avanti per sistemi triangolari inferiori con 
#  termini noti multipli e stessa matrice dei coefficienti
#------------------------------------------------------------------------------
def lowertrisol(L, B):
    """
    lowertrisol - Soluz. di sist. triang. inf. non sing. (per colonne) con piu' termini noti
    Soluzione simultanea di uno o piu' sistemi lineari con la stessa matrice L dei 
    coefficienti, triangolare inferiore non singolare. I termini noti (uno o piu')
    sono le colonne della matrice B.
    SYNOPSIS
      XS = lowertrisol(L, B)
    INPUT
      L  (float array) - Matrice triangolare inferiore non singolare
      B  (float array) - Vettore o matrice dei termini noti. La j-esima colonna
                         contiene il termine noto del j-esimo sistema lineare da 
                         risolvere, con matrice dei coefficienti sempre L
    OUTPUT
      XS (float array) - Vettore o matrice contenente, in ciascuna j-esima 
                         colonna, la soluzione del j-esimo sistema lineare.
    """
    # Controllo dell'input
    if ( (L is None) | (B is None) | (not L.size) | (not B.size) ):
        raise ValueError("Matrice e termini noti devono essere array non vuoti")
    mL, nL = L.shape; mB = B.shape[0];  eps = np.finfo(np.float64).eps
    # Breve controllo delle dimensioni dell'input
    if ( mL != nL ):
        raise ValueError("Matrice di coefficienti non quadrata")
    elif ( any( abs( np.diag(L) ) < eps * norm(L, np.inf) ) ):
        raise ValueError("Almeno un elemento diagonale di L e' numericamente troppo vicino a zero")
    elif (mL != mB):
        raise ValueError("I numeri di righe della matrice e dei termini noti devono coincidere")
    # Calcolo della/e soluzione/i
    XS = B.copy(); vecflag = False
    if ( mB == B.size ): # B is just a vector
        XS = XS[:, None]; vecflag = True
    XS[0, ] = XS[0, ] / L[0, 0]
    for j in range(1, mL):
        # operazione di prodotto esterno (diade) e somma di matrici (tipo BLAS2)
        XS[j:, ] = XS[j:, ] - np.outer( L[j:, j-1], XS[j-1, ] )
        XS[j, ]  = XS[j, ] / L[j, j]
    if ( vecflag ): XS = XS.flatten() # make XS a vactor, just like B
    return XS
# fine della funzione lowertrisol

#------------------------------------------------------------------------------
#  Algoritmo di sostituzione all'indietro per sistemi triangolari superiori
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
#  Soluzione di un sistema lineare mediante il metodo iterativo SOR
#------------------------------------------------------------------------------
def sor(A, b, x, maxit, tol, omega):
    """
    SOR - Metodo SOR (Gauss-Seidel estrapolato)
      Calcola un'approssimazione della soluzione xs del sistema Ax = b
      usando il metodo iterativo SOR con tolleranza d'arresto tol sulla norma
      infinito della differenza di due iterati successivi, partendo dal punto
      iniziale x e con un massimo maxit di iterazioni. Il vettore inziale
      verra' sovrascritto in uscita con l'approssimazione calcolata della
      soluzione.
      Il calcolo viene effettuato SENZA CALCOLO ESPLICITO E MEMORIZZAZIONE
      della matrice di iterazione, ma usando semplicemente gli elementi di A.
      Versione compatibile anche con matrice A in forma sparsa
    SYNOPSIS:
      [x, k] = sor(A, b, x, maxit, tol, omega)
    INPUT
      A     (doble array)   - Matrice (quadrata) del sistema lineare
      b     (double vector) - Vettore dei termini noti del sistema lineare
      x     (double vector) - Iterato iniziale (sovrascritto da xs in uscita)
      maxit (integer)       - Numero massimo di iterazioni consentite
      tol   (double)        - Tolleranza del criterio di arresto
      omega (double)        - Parametro di rilassamento del metodo SOR
    OUTPUT
      x     (double vector) - Approssimazione della soluzione
      k     (integer)       - Numero di iterazioni effettivamente compiute
    """
    # Inserire controlli sull'input...
    n = max(A.shape); k = 0; stop = False
    while (not stop):
        k += 1; xtemp = x.copy()
        for i in range(n):
            sum_ax = A[i, :i] @ x[:i] + A[i, i+1:] @ x[i+1:]
            x[i] = (b[i] - sum_ax) / A[i, i]
            x[i] = (1 - omega) * xtemp[i] + omega * x[i]
        stop = ( norm(xtemp - x, np.inf) < tol * norm(x, np.inf) ) \
               or ( k == maxit )
    if ( k == maxit ):
        print(f"WARNING: raggiunto il numero massimo di iterazioni maxit = {maxit:d}")
    return x, k
# fine della funzione sor

#------------------------------------------------------------------------------
#  Calcolo dei coefficienti diagonali della tabella delle differenze divise
#------------------------------------------------------------------------------
def tabdiff(x, y):
    """"
    tabdiff - Tabella delle differenze divise sui nodi x e i valori y
    INPUT
      x  (float array) - vettore dei nodi o punti di osservazione
      y  (float array) - vettore delle osservazioni
    OUTPUT
      d  (float array) - coefficienti del polinomio di Newton (ordinati
                         per grado crescente del termine corrispondente)
    """
    x = x.flatten(); y = y.flatten()
    n = x.size  # numero dei punti di interpolazione ( = grado polinomio + 1 )
    d = y.copy()
    for k in range(1, n):
        d[k:] = ( d[k:] - d[(k-1):(n-1)] ) / ( x[k:] - x[:(n-k)] )
    return d
# fine della funzione tabdiff
