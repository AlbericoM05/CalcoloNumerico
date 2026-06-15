import numpy as np
from time import time
from math import sin, cos
from os import system

system("clear")

def myFun(alpha, n):
    z = np.zeros(2)         # Array risultati
    t = z.copy()            # Array tempi

    tSTR = time()

    for k in range(0, n, 2):
        z[0] += (sin(k * alpha) + cos((k+1) * alpha))
    if not(n % 2):
        z[0] += sin(n * alpha)

    t[0] = time()

    z[1] = sum(np.sin(np.arange(0, n+1, 2) * alpha)) + sum(np.cos(np.arange(1, n+1, 2) * alpha))

    t[1] = time() - t[0]
    t[0] -= tSTR

    return z, t

alpha = 0.15
n = 500000

z, t = myFun(alpha, n)

print(f"Tipo Calcolo\t| Risultato\t| Tempo")
print("- - - - - - - - - - - - - - - - - - - - - - -")
print(f"Ciclo FOR\t| {z[0]:.3e}\t| {t[0]:.2f} sec.")
print(f"Vettore\t\t| {z[1]:.3e}\t| {t[1]:.2f} sec.")