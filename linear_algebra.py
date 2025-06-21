import numpy as np

# Matrix multiplication

a = np.ones((2, 3))
print(a)
b = np.full((3, 2), 2)
print(b)

c = np.matmul(a, b)  # or equivalently c = a @ b
print(c)

# Determinant

c = np.identity(3)
print(c)
det = np.linalg.det(c)
print(det)