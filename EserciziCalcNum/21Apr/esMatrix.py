# Alberico Moretto
# 203799

# Librerie
import numpy as np

# Esercizio
A = np.array([
    [1., -1., 2., 0.],
    [-3., 2., 0., 4],
    [0., -5., 1., -2.],
    [0., 0., 2., 1.]
], dtype = np.float64)
b = np.array([0., -2., 1., 0.], dtype = np.float64)

m, n = A.shape; I4 = np.eye( b.size )

P1 = np.eye( m ); P1[ [0, 1], : ] = P1[ [1, 0], : ]
A1 = np.column_stack( (A, b) )
A1Max = -3.; ind1 = 1;
A1Perm = P1 @ A1
m1 = np.array( [0., 1., 0., 0.], dtype = np.float64 ) / A1Max

L1 = I4 - np.outer( m1, I4[0, :] )

A2 = L1 @ A1Perm

print(f"{A1}\n\n{P1}\n\n{A1Perm}\n\n{m1}\n\n{L1}")