import numpy as np
from scipy.linalg.blas import drotg
from scipy.linalg import norm
from libs.allegati_2019_06_12 import uppertrisol, lowertrisol, gausspivpar
from os import system

system("clear")

A = np.array([
    [4., -2., 1., 0.],
    [-6., 4., 5./2., 0.],
    [1., 5./2., -6., 1.],
    [0., 0., 1., 4.]
], dtype = np.float64)
b = np.array([0., 2., -6., 0.], dtype = np.float64)

L, R, p = gausspivpar(A)

xPy = uppertrisol(R, lowertrisol(L, b))
print(f"Soluzione: \n{xPy}")
print(f"Residuo normalizzato: \n{(b -A @ xPy) / norm(b, np.inf)}")

print("\n\nRotazioni di Givens:")
Ab = np.column_stack(A, b)
c, s = drotg(A[0, 0], A[1, 0])
G12 = np.eye(4)
G12[:2, :2] = np.array([[c, s], [-s, c]])
A1 = G12 @ Ab
print(f"G12:\n{G12}\nA1:\n{A1}")
c, s = drotg(A[0, 0], A[2, 0])
G13 = np.eye(4)
G13[:2, :2] = np.array([[c, s], [-s, c]])
A2 = G13 @ A1
print(f"G13:\n{G13}\nA2:\n{A2}")
c, s = drotg(A[1, 1], A[2, 1])
G23 = np.eye(4)
G23[:2, :2] = np.array([[c, s], [-s, c]])
A3 = G23 @ A2
print(f"G23:\n{G23}\nA3:\n{A3}")
c, s = drotg(A[2, 2], A[3, 2])
G34 = np.eye(4)
G34[:2, :2] = np.array([[c, s], [-s, c]])
A4 = G34 @ A3
print(f"G34:\n{G34}\nA4:\n{A4}")

R = np.triu(A4[:, :4])
Q = (G34 @ G23 @ G13 @ G12).T

print(f"R:\n{R}\nQ:\n{Q}")
