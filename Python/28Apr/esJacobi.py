import numpy as np
from metodiInterativi import jacobi
from metodiInterativi import gaussSeidel
from metodiInterativi import sor


A = np.array([[2, -1, 0],
              [-1, 3, 1],
              [0, 1, 5]], dtype = np.float32)
b = np.array([1, 3, 6], dtype = np.float32)
x = np.zeros(3, dtype = np.float32)

maxit = 100
tol = 1e-2

####################
# METODO DI JACOBI #
####################

print("\nMetodo di Jacobi\n")

xJ, xiter = jacobi(A, b, np.copy(x), maxit, tol)

print(f"Soluzione calcolata: {xJ}")
print(f"Iterazioni: {xiter}")

Dinv = np.diag(1.0 / np.diag(A))
L = np.tril(A, -1)
U = np.triu(A, 1)

J = Dinv @ (L + U)

eigJ = np.linalg.eigvals(J)
rhoJ = max(abs(eigJ))

print(f"Raggio spettrale della matrice di Jacobi: {rhoJ}")


RinfJ = -np.log(rhoJ)

print(f"Velocità asintotica di convergenza: {RinfJ}")

##########################
# METODO DI GAUSS SEIDEL #
##########################

print("\nMetodo di Gauss-Seidel\n")

xGS, xiter = sor(A, b, np.copy(x), maxit, tol, 1.2)

print(f"Soluzione calcolata: {xGS}")
print(f"Iterazioni: {xiter}")

Dinv = np.diag(1.0 / np.diag(A))

GS = np.linalg.inv(np.tril(A)) @ U

eigGS = np.linalg.eigvals(GS)
rhoGS = max(abs(eigGS))

print(f"Raggio spettrale della matrice di Jacobi: {rhoGS}")


RinfGS = -np.log(rhoGS)

print(f"Velocità asintotica di convergenza: {RinfGS}")

#################
# METODO DI SOR #
#################

print("\nMetodo di SOR\n")

xGS, xiter = gaussSeidel(A, b, np.copy(x), maxit, tol)

print(f"Soluzione calcolata: {xGS}")
print(f"Iterazioni: {xiter}")

Dinv = np.diag(1.0 / np.diag(A))

GS = np.linalg.inv(np.tril(A)) @ U

eigGS = np.linalg.eigvals(GS)
rhoGS = max(abs(eigGS))

print(f"Raggio spettrale della matrice di Jacobi: {rhoGS}")


RinfGS = -np.log(rhoGS)

print(f"Velocità asintotica di convergenza: {RinfGS}")