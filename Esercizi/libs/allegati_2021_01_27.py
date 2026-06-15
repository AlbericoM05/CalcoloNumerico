#------------------------------------------------------------------------------
#  Codici allegati alla prova scritta del 27/01/2021 - Versione Python
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
#  Algoritmo di sostituzione all'indietro per sistemi triangolari superiori con 
#  termini noti multipli e stessa matrice dei coefficienti
#------------------------------------------------------------------------------
def uppertrisol(R, B):
    """
    uppertrisol - Soluz. di sist. triang. sup. non sing. (per colonne) con piu' termini noti
    Soluzione simultanea di uno o piu' sistemi lineari con la stessa matrice R dei 
    coefficienti, triangolare superiore non singolare. I termini noti (uno o piu')
    sono le colonne della matrice B.
    SYNOPSIS
      XS = uppertrisol(R, B)
    INPUT
      R  (float array) - Matrice triangolare superiore non singolare
      B  (float array) - Vettore o matrice dei termini noti. La j-esima colonna
                         contiene il termine noto del j-esimo sistema lineare da 
                         risolvere, con matrice dei coefficienti sempre R
    OUTPUT
      XS (float array) - Vettore o matrice contenente, in ciascuna j-esima 
                         colonna, la soluzione del j-esimo sistema lineare.
    """
    # Controllo dell'input
    if ( (R is None) | (B is None) | (not R.size) | (not B.size) ):
        raise ValueError("Matrice e termini noti devono essere array non vuoti")
    mR, nR = R.shape; mB = B.shape[0];  eps = np.finfo(np.float64).eps
    # Breve controllo delle dimensioni dell'input
    if ( mR != nR ):
        raise ValueError("Matrice di coefficienti non quadrata")
    elif ( any( abs( np.diag(R) ) < eps * norm(R, np.inf) ) ):
        raise ValueError("Almeno un elemento diagonale di R e' numericamente troppo vicino a zero")
    elif (mR != mB):
        raise ValueError("I numeri di righe della matrice e dei termini noti devono coincidere")
    # Calcolo della/e soluzione/i
    XS = B.copy(); nR -= 1; vecflag = False
    if ( mB == B.size ): # B is just a vector
        XS = XS[:, None]; vecflag = True
    XS[nR, ] = XS[nR, ] / R[nR, nR]
    for j in range(nR-1, -1, -1):
        # operazione di prodotto esterno (diade) e somma di matrici (tipo BLAS2)
        XS[0 : j+1, ] = XS[0 : j+1, ] - np.outer( R[0 : j+1, j+1], XS[j+1, ] )
        XS[j, ]  = XS[j, ] / R[j, j]
    if ( vecflag ): XS = XS.flatten() # make XS a vactor, just like B
    return XS
# fine della funzione uppertrisol

#------------------------------------------------------------------------------
#  Fattorizzazione LR con metodo di Gauss con pivoting totale
#------------------------------------------------------------------------------
def gausspivtot( A ):
    """
    gausspivtot - Algoritmo di eliminazione di Gauss con pivoting totale (non sovrascrive A)
    Restituisce i fattori L (triangolare inferiore a diagonale unitaria), R (tringolare 
    superiore) ed i vettori p di permutazione per righe e q di permutazione per colonne
    della fattorizzazione PAQ = LR con il metodo di Gauss con pivoting totale.
    Implementazione applicabile anche al caso di matrice A non quadrata.
    SINOPSYS
        L, R, p, q = gausspivtot( A )
    INPUT
        A (float array)   - Matrice da fattorizzare
    OUTPUT
        L (float array)   - Matrice L triangolare (o trapez.) inferiore a diagonale unitaria
        R (float array)   - Matrice R triangolare (o trapezoidale) superiore
        p (integer array) - Vettore delle permutazioni di righe
        q (integer array) - Vettore delle permutazioni di colonne
    """
    m, n = A.shape; tol = np.finfo( np.float64 ).eps * norm(A, np.inf)
    A = np.float64( A.copy() ) # si effettua una copia per non sovrascrivere A  
    p = np.arange( m ); q = np.arange( n )
    for k in range( min(m-1, n) ):
        # indici del primo elemento di modulo massimo nella sottomatrice di A del passo k
        i, j = np.unravel_index( abs( A[k:, k:] ).argmax(), (m-k, n-k) ); i += k; j += k
        if ( i != k ): # scambio delle righe k-esima e i-esima e dei corrisp. indici in p
            A[ [k, i], : ] = A[ [i, k], : ]; p[ [k, i] ] = p[ [i, k] ]
        if ( j != k ): # scambio delle colonne k-esima e j-esima e dei corrisp. indici in q
            A[ :, [k, j] ] = A[ :, [j, k] ]; q[ [k, j] ] = q[ [j, k] ]
        if ( abs( A[k, k] ) > tol ):
            A[ (k+1): , k] /= A[k, k]
            # operazione di base di livello 2: aggiornamento mediante diade
            A[ (k+1): , (k+1): ] -= np.outer( A[ (k+1): , k ], A[ k, (k+1): ] )
        else:
            # se il pivot e' molto piccolo in modulo, non si aggiorna la sottomatrice 
            print(f"WARNING: elemento pivot di modulo molto piccolo (< tol = {tol})")
    # Si costruiscono in modo esplicito i fattori L ed R, estraendoli da A (poco efficiente!)
    nn = min(m,n); L = np.tril( A[:, :nn], -1 ); L[ range(nn), range(nn) ] = 1.0
    return L, np.triu( A[:nn, :] ), p, q
# fine della funzione gausspivtot
