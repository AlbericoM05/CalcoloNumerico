import numpy as np

p = np.array([3., -3., 6., 11., np.pi, 0., -2.], dtype = np.float64)
x0 = np.float64(input("Inserisci un punto x0:"))

q, r = ruffiniHorner(p, x0)
q1, r1 = ruffiniHorner(q, x0)
q2, r2 = ruffiniHorner(q1, x0)

print("/ / / / / / | Valore")
print(f"p(x0)\t| {r:.3f}")
print(f"p'(x0)\t| {r1:.3f}")
print(f"p'\'(x0)\t| {2 * r2:.3f}")
