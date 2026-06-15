# Moretto Alberico
# 203799

import numpy as np
from scipy.linalg import qr, lu, norm
from libs.sollower import sollower
from libs.solupper import solupper
from os import system

def miaDiade(r, s):
    if (r.size == 0 or s.size == 0):
        raise ValueError("I vettori sono vuoti!")
    
    A = np.outer(r.flatten(), s.flatten())                                                      # Una diade si crea moltiplicando una riga per una colonna, quindi traspformo 's' in una riga.
    m, n = A.shape                                                                              # Dimensioni della diade
    Q, R = qr(A)                                                                                # 
    normaInfDiade = norm(A, np.inf)                                                             # Norma infinita della diade (serve per calcolare il rango)
    diagDiade = np.diag(R)                                                                      # Diagonale della diade (serve per calcolare il rango)
    rangoDiade = np.count_nonzero(abs(diagDiade) > np.finfo(np.float64).eps * normaInfDiade)
    detDiade = np.prod(diagDiade)

    return rangoDiade, detDiade, normaInfDiade

system("clear")

A = np.array([
    [1., 0., -1., 0.],
    [0., 3., 0., 1.],
    [1., 0., 2., 0.],
    [0., 1., 0., 2.]
], dtype = np.float64)
b  = np.array([0., -1., 0., -1.], dtype = np.float64)

Q, R = qr(A)
xQR = solupper(R, np.transpose(Q) @ b)
P, L, U = lu(A)
xLU = solupper(U, sollower(L, np.transpose(P) @ b))

normaDiffSol = norm(xQR - xLU, np.inf)
normaRelDiffSol = normaDiffSol / norm(xQR, np.inf)

print("Norma infinito della differenza delle soluzioni:")
print(f"Assoulta: {normaDiffSol:.3e}\t|\tRelativa: {normaRelDiffSol:.3e}")

rango, det, norma = miaDiade(Q[:, 0], R[0, :])
print(f"{rango = }\t|\t{det = }\t|\t{norma = }")
