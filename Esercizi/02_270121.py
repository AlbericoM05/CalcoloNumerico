# Moretto Alberico
# 203799

import numpy as np
import matplotlib.pyplot as plt

from time import time
from os import system

def myFun(x, s, w):
    indx = np.where((x > -s) and ((x * (np.log(x + s) - w)) > 0))
    z = np.zeros(indx.size)

    N = 10000
    tm = time()

    for k in range (N):
        z = (x * (np.log(x + s) - w)) ** 0.25

    tm = time() - tm / np.float64(N)

    return z, indx, tm

system("clear")

a = 0.5
b = 5
