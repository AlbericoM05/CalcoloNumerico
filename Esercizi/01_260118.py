# Moretto Alberico
# 203799

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from os import system

system("clear")

p6 = np.array([3., -9., 0., 11., -pi, 0., -2.], dtype = np.float64)

alpha = np.float64(input("\nInserisci alpha: "))

valP = np.polyval(p6, alpha)
dp = np.polyder(p6, alpha)
sp = np.polyder(dp, alpha)
valDP = np.polyval(dp, alpha)
valSP = np.polyval(sp, alpha)

print(f"Valore polinomio in alpha = {alpha:.4g}: p(alpha) = {valP:.5f}")
print(f"Valore derivata prima in alpha = {alpha:.4g}: p'(alpha) = {valDP:.5f}")
print(f"Valore derivata seconda in alpha = {alpha:.4g}: p'\'(alpha) = {valSP:.5f}")

x = np.linspace(0.5, 2)
plt.plot(x, np.polyval(p6, x))
plt.show()