import numpy as np

def miaDiade(r, s):
    """
    Calcola la diade rs' e ne ricava il rango, determinante e norma infinito.
    """

    if r.size == 0 or s.size == 0:
        raise ValueError("I due vettori in input non possono essere vuoti!")
    
    if not (r.shape[1] == 1 and s.shape[1] == 1):
        raise ValueError("I due vettori devono essere vettori-colonna")

    # '@' sta per 'prodotto tra vettori'
    A = r @ np.transpose(s) # A è una diade
    m, n = A.shape
    Q, R = np.linalg.qr(A)
    diagR = np.diag(R)

    normaInf = np.linalg.norm(A, np.inf)
    rango = len(np.argwhere(abs(diagR) > np.finfo(float).esp * normaInf))
    determinante = np.prod(diagR)

    return rango, determinante, normaInf