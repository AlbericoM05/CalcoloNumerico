import numpy as np
from scipy.linalg import lu, norm


A = np.array([
    [1, 3, -1],
    [1, 2, 0.5],
    [5, 10, -1]
], dtype = np.float64)
b = np.array([-2, 0.5, -1], dtype = np.float64)

R, L, p = gausspivpar(A)

# Calcolo e stampa della soluzione
sol = solupper(R, sollower(L, b[p]))
print(f"Soluzione: {sol}")

# Calcolo e stampa del residuo normalizzato
resNorm = norm(b - A @ sol, np.inf) / norm(b, np.inf)
print(f"Residuo normalizzato: {resNorm}")

p1, L1, R1 = lu(A, p_indices=True)
print(f"max( |L - L1| ) = {max(abs(L - L1).flatten())}")
print(f"max( |R - R1| ) = {max(abs(R - R1).flatten())}")
print(f"{p = } {p1 = }")