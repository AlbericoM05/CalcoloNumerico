import numpy as np
from scipy.linalg import cholesky, norm
from libs.allegati_2021_01_27 import gausspivtot, uppertrisol, lowertrisol
from os import system

system("clear")

A = np.array(np.float64([
    [1., -1., 2., 0.],
    [-3., 2., 0., 4.],
    [0., -5., 1., -2.],
    [0., 0., 2., 1.]
]))
b = np.array([0., -2., 1., 0.], dtype = np.float64)
xTeoria = np.array([-14., 8., 11., -22.], dtype = np.float64) / 15.0

L, R, p, q = gausspivtot(A)
Q = np.eye(b.size)
Q = Q[:, p]

xPy = uppertrisol(R, lowertrisol(L, b))
xPy = Q @ xPy
print(f"L: \n{L}\nR: \n{R}\np: \n{p}\nSoluzione: \n{xPy}\n")
print("Massima differenza in modulo dalla soluzione teorica: ")
print(f"{norm(xPy - xTeoria, np.inf)}")
print("Residuo normalizzato: ")
print(f"{(b - A @ xPy) / norm(b, np.inf)}")

# Residuo normalizzato (inf)


B = A.T @ A
print("\n\nFattorizzazione di Cholesky")
try:
    LChol = cholesky(B, lower=True)
    c = np.array([7./3., 1., 2., -5./6.], dtype = np.float64)
    print(f"B è definita positiva\nLChol: \n{LChol}")
    xChol = uppertrisol(LChol.T, lowertrisol(LChol, c))
    print(f"Soluzione con Cholesky: {xChol}")
    print(f"{np.allclose(c, B @ xChol) = }")
except np.LinAlgError:
    print("ATTENZIONE! B non è definita positiva!")
