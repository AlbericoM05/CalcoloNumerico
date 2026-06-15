# Moretto Alberico
# 203799

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from os import system

def myFun(b, p, q, x):
    if any([(arg is None) or np.isnan(arg).any() or np.isinf(arg).any() for arg in [b, p, q, x]]):
        raise ValueError("Valori non validi!")
    
    fc = np.zeros(len(x))
    fv = fc

    for i in range (len(x)):
        fc[i] = - sin(p[0] * b ** (q[0] * x[i])) + cos(p[1] * b ** (q[1] * x[i])) - sin(p[2] * b ** (q[2] * x[i])) + cos(p[3] * b ** (q[3] * x[i]))

    fv = - np.sin(p[0] * b ** (q[0] * x)) + np.cos(p[1] * b ** (q[1] * x)) - np.sin(p[2] * b ** (q[2] * x)) + np.cos(p[3] * b ** (q[3] * x))

    return fc, fv

system("clear")

b = np.float64(0.7)
p = np.array([-2, 1., 3., -1], dtype = np.float64)
q = np.array([5., 4., 3., 2.], dtype = np.float64)
x = np.array([1.7e-7, 1.7e-6, 1.7e-5, 1.7e-4, 1.7e-3, 1.7e-2, 1.7e-1, 1.7e+00, 1.7e+1, 1.7e+2], dtype = np.float64)

fc, fv = myFun(b, p, q, x)

print("- - - - - - - - - - - - - - - - - - - - - - -")
print("Tabella di f(x_i)")
print("- - - - - - - - - - - - - - - - - - - - - - -")
for i in range (len(x)):
    print(f"X_{i} = {x[i]:1.1e}\t|{fc[i]:11.4f}\t|{fv[i]:11.4f}")

plt.plot(x, fv)
plt.show()