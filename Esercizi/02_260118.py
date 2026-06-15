# Moretto Alberico
# 203799

import numpy as np
from time import time
from math import isnan, isinf, cos, sin
from os import system

system("clear")

def myFun(alpha, n):
    # Controllo input con costrutto figo
    if any([((arg is None) or isnan(arg) or isinf(arg)) for arg in [n, alpha]]):
        raise ValueError("Errore: 'n' o 'alpha' sono valori non validi!")
    if((n < 1) or int(n) != n):
        raise ValueError("Errore: 'n' non è un intero!")
    
    z = np.zeros(2)
    t = z.copy()
    tstart = time()

    for k in range(0, n, 2):
        z[0] = z[0] + (sin(k * alpha) + cos((k+1) * alpha))
    if(not(n % 2)):
        z[0] = z[0] + sin(n * alpha)
    
    t[0] = time()

    z[1] = sum(np.sin(np.arange(0, n+1, 2) * alpha)) + sum(np.cos(np.arange(1, n+1, 2) * alpha))

    t[1] = time() - t[0]
    t[0] -= tstart

    return z, t

alpha = np.float64(input("Inserire alpha: "))
n = int(input("Inserisci n: "))

z, t = myFun(alpha, n)

for i in range (2):
    print(f"{i+1:d}) f({alpha}) = {z[i]: 9.3e},\t Tempo: {t[i]:9.3e} sec.")