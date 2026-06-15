# Moretto Alberico
# 203799

import numpy as np
import matplotlib.pyplot as plt
from os import system
from time import sleep
from math import isnan, isinf

def myFun(k, a, b, N):
    if any([(arg is None) or isnan(arg) or isinf(arg) for arg in [k, a, b, N]]):
        raise ValueError("Errore, i valori in input non sono validi!")
    elif (k < 1) or (k > 10):
        raise ValueError("Errore, k non rientra nell'intervallo accettato!")
    elif (a >= b):
        raise ValueError("Errore, a deve essere minore stretto di b!")
    elif (N <= 0):
        raise ValueError("Errore, n deve essere maggiore stretto di 0!")
    
    xx = np.linspace(a, b, N + 1)
    return (k * (1 - xx) ** 2) * np.exp(-xx * xx)

system("clear")

a = -3; b = -a; n = 50
x = np.linspace(a, b, n + 1)

for k in np.arange(1, 10, 0.1):
    y = myFun(k, a, b, n)

    plt.plot(x, y)
    plt.ylim(-1, 20)
    plt.xlabel("Intervallo di campionamento")
    plt.ylabel("Valori della funzione")
    plt.title(f"Grafico della funzione f_k(x) per k = {k:4.1f}")
    plt.show()

    sleep(0.05)