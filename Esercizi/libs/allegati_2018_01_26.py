#------------------------------------------------------------------------------
#  Codici allegati alla prova scritta del 26/01/2018 - Versione Python
#------------------------------------------------------------------------------
import numpy as np
import scipy.sparse as sp
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
    m, n = L.shape; eps = np.finfo(np.float64).eps
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
        XS[j:, ] = XS[j:, ] - np.outer( L[j:, j-1], XS[j-1, ] )
        # operazione di prodotto esterno (diade) e somma di matrici (tipo BLAS2)
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
    m, n = R.shape; eps = np.finfo(np.float64).eps
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
#  Fattorizzazione LU con metodo di Gauss con pivoting parziale
#------------------------------------------------------------------------------
def gausspivpar( A ):
    """
    gausspivpar - Algoritmo di eliminazione di Gauss con pivoting parziale (non sovrascrive A)
    Restituisce i fattori L (triangolare inferiore a diagonale unitaria), 
    R (tringolare superiore) ed il vettore p di permutazione per righe della 
    fattorizzazione PA = LR con il metodo di Gauss con pivoting parziale
    Implementazione applicabile anche al caso di matrice A non quadrata.
    SINOPSYS
        L, R, p = gausspivpar( A )
    INPUT
        A (float array)   - Matrice da fattorizzare
    OUTPUT
        L (float array)   - Matrice L triangolare (o trapez.) inferiore a diagonale unitaria
        R (float array)   - Matrice R triangolare (o trapezoidale) superiore
        p (integer array) - Vettore delle permutazioni di righe
    """
    m, n = A.shape; tol = np.finfo( np.float64 ).eps * norm(A, np.inf)
    A = np.float64( A.copy() ) # si effettua una copia per non sovrascrivere A  
    p = np.arange( m )
    for k in range( min(m-1, n) ):
        # indice dell'elemento sottodiagonale di modulo massimo (argmax() ritorna solo il primo)
        i = abs( A[k:, k] ).argmax() + k
        if ( i != k ): # scambio delle righe k-esima e i-esima e dei corrisp. indici in p
            A[ [k, i], : ] = A[ [i, k], : ]; p[ [k, i] ] = p[ [i, k] ]
        if ( abs( A[k, k] ) > tol ):
            A[ (k+1): , k] /= A[k, k]
            # operazione di base di livello 2: aggiornamento mediante diade
            A[ (k+1): , (k+1): ] -= np.outer( A[ (k+1): , k ], A[ k, (k+1): ] )
        else:
            # se il pivot e' molto piccolo in modulo, non si aggiorna la sottomatrice 
            print(f"WARNING: elemento pivot di modulo molto piccolo (< tol = {tol})")
    # Si costruiscono in modo esplicito i fattori L ed R, estraendoli da A (poco efficiente!)
    nn = min(m, n); L = np.tril( A[:, :nn], -1 ); L[ range(nn), range(nn) ] = 1.0
    return L, np.triu( A[:nn, :] ), p
# fine della funzione gausspivpar

#------------------------------------------------------------------------------
#  Soluzione di un sistema lineare mediante il metodo iterativo di Jacobi
#------------------------------------------------------------------------------
def jacobi(A, b, x, maxit, tol):
    # jacobi - Metodo iterativo di Jacobi per sistemi lineari
    # ATTENZIONE: sovrascrive il vettore b con b / diag(A) e l'iterato iniziale
    # x con l'approssimazione della soluzione del sistema A x = b.
    # Versione compatibile anche con matrice A in forma sparsa
    m, n = A.shape
    if ( m != n ):
        raise ValueError("matrice dei coefficienti non quadrata.")
    if ( sp.issparse(A) ): d = A.diagonal()
    else: d = np.diag( A )
    if ( np.any( np.abs(d) < np.finfo(float).eps * norm(d, np.inf) ) ):
        raise ValueError("elementi diagonali troppo piccoli.")
    b /= d; k = 0; stop = False
    while ( not stop ):
        k += 1; xtemp = x.copy() # serve per la condizione di arresto
        x = x - ( ( A @ x ) / d ) + b # istruzione vettoriale
        stop = ( norm(xtemp - x, np.inf) < tol * norm(x, np.inf) ) \
               or ( k == maxit )
    if ( k == maxit ):
        print(f"WARNING: raggiunto il numero massimo di iterazioni maxit = {maxit:d}")
    return x, k
# fine della funzione jacobi

#------------------------------------------------------------------------------
#  Coefficienti e valutaz. del polinomio interpolante nella forma di Lagrange
#------------------------------------------------------------------------------
def polyLagrange(x, y, punti):
    """
    polyLagrange - Polinomio interpolante nella forma di Lagrange
    INPUT
      x     (float array) - vettore dei nodi o punti di osservazione
      y     (float array) - vettore delle osservazioni
      punti (float array) - vettore dei punti in cui calcolare il polinomio di Lagrange
    OUTPUT
      p     (float array) - valore del polinomio nel vettore punti
      coeff (float array) - coefficienti della base di Lagrange
    """
    
    n1 = y.size; coeff = np.zeros( x.size ); p = np.zeros( punti.size )
    for k in range( n1 ):
        idxk = np.hstack( ( np.arange(k, dtype = np.int32), np.arange(k+1, n1, dtype = np.int32) ) )
        coeff[k] = y[k] / ( x[k] - x[idxk] ).prod()
    for j in range( punti.size ):
        ij = ( punti[j] == x ).nonzero()
        if ( (ij is None) or (not ij[0].size) ):
            # calcolo del valore del polinomio di Lagrange in punti[j]
            p[j] = ( punti[j] - x ).prod() * sum( coeff / (punti[j] - x) )
        else:
            # punti[j] e' parte del vettore dei nodi: si copia il corrispondente y
            p[j] = y[ ij[0][0] ]
    return p, coeff
# fine della funzione polyLagrange

#------------------------------------------------------------------------------
#  Metodo di Newton-Raphson per la ricerca di zeri di funzioni scalari
#------------------------------------------------------------------------------
def newtonRaphson(fname, fpname, x0, tolx, tolf, maxit):
    """
    newtonRaphson - Metodo di Newton-Raphson per la ricerca di uno zero di una funzione scalare 
    SYNOPSIS
      x, it = newtonRaphson(fname, fpname, x0, tolx, tolf, maxit)
    INPUT
      fname  (callable) - Function della funzione
      fpname (callable) - Function della derivata prima 
      x0     (float)    - Stima iniziale
      tolx   (float)    - Distanza minima fra iterati successivi
      tolf   (float)    - Soglia verso zero dei valori di f(x)
      maxit  (integer)  - Numero massimo di interazioni 
    OUTPUT
      x      (float)    - Approssimazione della soluzione
      it     (integer)  - Numero di iterazioni eseguite
    """
    tolfp = min( tolf, 10 * np.finfo( np.float64 ).eps )
    # Metodo di Newton-Raphsopn
    x = x0;  fx = fname( x );  it = 0;  stop = ( abs(fx) < tolf )
    while ( not stop ):
        it += 1;  fpx = fpname( x )
        if ( abs(fpx) < tolfp ): raise ValueError( f"|f(xk)| (k = {it}) troppo piccolo" )
        d = fx / fpx;  x -= d;  fx = fname( x )
        stop = ( (abs(fx) < tolf and abs(d) < tolx*abs(x)) \
                 or ( not fx ) or ( it == maxit ) )
    if ( it == maxit ):
        print( "\nWARNING: raggiunto il massimo numero di iterazioni\n" )
    return x, it
# fine della funzione newtonRaphson 
