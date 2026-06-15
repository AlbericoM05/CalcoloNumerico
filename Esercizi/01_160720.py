# Moretto Alberico
# 203799

from os import system
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, log, exp, log10, sqrt, pi, tan

system('clear')

print("Esercizio 1")

p7 = np.zeros(8)

p7[0] = tan(1.37 * pi ** 2- exp(-0.6))
p7[1] = -log10(4.2 + sin(0.77 * exp(1.3)))
p7[3] = abs(sqrt(5.7E-2) + cos(-3 * pi / 5))
p7[5] = 5 * cos(sin(12.3 - pi ** (2.1)))
p7[7] = log(3.1E-4)

alpha = np.float64(input("\nInserisci un alpha: "))

valorePol = np.polyval(p7, alpha)
derp = np.polyder(p7, alpha)
ders = np.polyder(derp, alpha)
valoreDerp = np.polyval(derp, alpha)
valoreDers = np.polyval(ders, alpha)

print(f"Valore del polinomio in alpha = {alpha:.4g}: p(alpha) = {valorePol:.5f}")
print(f"Valore della derivata prima in alpha = {alpha:.4g}: p'(alpha) = {valoreDerp:.5f}")
print(f"Valore della derivata seconda in alpha = {alpha:.4g}: p'\'(alpha) = {valoreDers:.5f}")

x = np.linspace (-1.7, 2.0, 201)
plt.plot(x, np.polyval(p7, x))
plt.show()