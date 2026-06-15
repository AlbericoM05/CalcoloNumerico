# Moretto Alberico
# 203799

import numpy as np
import matplotlib.pyplot as plt
from os import system

system("clear")

p6 = np.array(np.float64([-np.exp(-np.pi), 1.4, 1, 0., -np.log(207.13), 0, 3*np.pi]))

alpha = np.float64(input("\nInserisci alpha: "))

valP = np.polyval(p6, alpha)
dp = np.polyder(p6, alpha)
sp = np.polyder(dp, alpha)
valDP = np.polyval(dp, alpha)
valSP = np.polyval(sp, alpha)

print(f"Valore polinomio in alpha = {alpha:.4g}: p(alpha) = {valP:.5f}")
print(f"Valore derivata prima in alpha = {alpha:.4g}: p'(alpha) = {valDP:.5f}")
print(f"Valore derivata seconda in alpha = {alpha:.4g}: p'\'(alpha) = {2*valSP:.5f}")

x = np.linspace(0.8, 1.7, 201)
plt.plot(x, np.polyval(p6, x))
plt.show()