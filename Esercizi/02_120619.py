# Moretto Alberico
# 203799

import numpy as np
import matplotlib.pyplot as plt
from os import system
from time import time

def myFun(x, alpha, beta):
    return np.cos(x) / (alpha - np.exp(beta * x))

system("clear")

a = -1.5; b = 2.3
alpha = 2; beta = 1.2
N = 2001

x = np.linspace(a, b, N)
fx = np.zeros(N)
z = fx

timestart = time()

for i in range (N):
    fx[i] = myFun(x[i], alpha, beta)
tempoCiclo = time()

z = myFun(x, alpha, beta)

tempoVett = time() - tempoCiclo
tempoCiclo -= timestart

print(f"Tempo impiegato dal ciclo: {tempoCiclo:.4e}")
print(f"Tempo impiegato dalla funzione con il vettore: {tempoVett:.4e}")

diffRel = abs(tempoCiclo - tempoVett) / abs(tempoCiclo)
diffPer = abs(tempoVett) * 100 / abs(tempoCiclo)

print(f"Differenza relativa di tempo: {diffRel:.4e}")
print(f"Differenza percentuale di tempo: {diffPer:.2f}%")

plt.plot(x, z); plt.show()