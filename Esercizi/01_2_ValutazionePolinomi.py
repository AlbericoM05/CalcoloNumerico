import numpy as np

coef = [1, 0, -6, 0, 8, -7]
alpha = -2

ris_py = np.polyval(coef, alpha)

ris_der1 = np.polyder(coef, 1)
ris_der2 = np.polyder(coef, 2)
ris_der3 = np.polyder(coef, 3)
ris_der1 = np.polyval(ris_der1, alpha)
ris_der2 = np.polyval(ris_der2, alpha)
ris_der3 = np.polyval(ris_der3, alpha)

print(f"Il valore del polinomio calcolato da Python in alpha = {alpha} è: {ris_py}")
print(f"I valori delle derivate 1^, 2^ e 3^ del polinomio calcolato da Python in alpha = {alpha} sono: {ris_der1}, {ris_der2}, {ris_der3}")