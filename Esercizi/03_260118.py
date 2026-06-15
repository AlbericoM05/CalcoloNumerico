# Moretto Alberico
# 203799

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu, norm
from libs.sollower import sollower
from libs.solupper import solupper

A = np.array([
    [1.0, 3.0, -1.0],
    [1.0, 2.0, 0.5],
    [5.0, 10.0, -1.0]
], dtype = np.float64)
b = np.array([-2.0, 0.5, -1.0], dtype = np.float64)
L, R, p = gausspivpar(A.copy())

x1 = solupper(R, sollower(L, b[p]))
print("Soluzione:"); print(x1)

print(f"\nResiduo normalizzato = {norm(b - A @ x1, np.inf) / norm(b, np.inf)}")
print("p1, L1, R1 = lu(A, p_indices = True)")

p1, L1, R1 = lu(A, p_indices = True)

print(f"max(|L - L1|) = {max(abs(L - L1).flatten())}")
print(f"max(|R - R1|) = {max(abs(R - R1).flatten())}")
print(f"{p = } {p1 = }")

print(f"{np.allclose(A[p, :], L @ R) = }")
print(f"{np.allclose(A, L1[p1, :] @ R) = }")

n = 30
normInfSol = np.zeros(n)
normInfSol[0] = norm(x1, np.inf)
c = b[p]
kind = p[1]

for k in range(1, n):
    c[kind] = 0.5 * c[kind]
    normInfSol[k] = norm(solupper(R, sollower(L, c)), np.inf)

plt.plot(normInfSol, 'bo')
plt.ylim(1.75, 3.25)
plt.xlabel("Valori del parametro k nel termine noto [-2, 2^{-k}, -1]")
plt.ylabel("Norma infinito della soluzione del sistema")
plt.title("Norma infinito della soluzione al variare di k")
plt.show()