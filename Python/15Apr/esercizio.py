import numpy as np
import scipy as sp

from diade import miaDiade
from sollower import sollower
from solupper import solupper

# Nel file caricato da Paparella ci sono anche i commenti.

A = np.array([1,1,1], [0,1,-1], [1,1,2])
b = np.array([3,0,4])

# Fattorizzazione QR di Givens
Q, R = np.linalg.qr(A);
# Ax = b -> QRx = b-> Rx = Q'b
# 1) Il calcolatore y = Q'b
# 2) Risolvo Rx = y

# Fattorizzazione LR con piv. parziale
y = np.transpose(Q) @ b
xQR = solupper(R, y)

P, L, U = sp.linalg.lu(A)
xLU = solupper(U, sollower(L, P @ b))

normaDiffSol = np.linalg.norm(xLU - xQR, np.inf)
normaRelDiffSol = normaDiffSol / np.linalg.norm(xQR, np.inf)

print(xLU)
print(xQR)
print(f"Norma inf. della differenza delle soluzioni: ", { normaDiffSol })
print(f"Norma inf. della differenza rel. delle soluzioni: ", { normaRelDiffSol })

# rankDiade, detDiade, normDiade = miaDiade(
#     np.transpose(np.atleast_2d)
# )
