import numpy as np
print(np.__version__)

# Array creation
a = np.array([1, 2, 3], dtype='int32')
print(a)

b = np.array([[9.0, 8.0, 7.0], [6, 5, 4]])
print(b)

# Array properties
print(f"Number of dimensions of a: {a.ndim}") # 1 dimension
print(f"Number of dimensions of b: {b.ndim}") # 2 dimensions
print(f"Shape of a: {a.shape}") # (3,)
print(f"Shape of b: {b.shape}") # (2, 3)
print(f"Data type of a: {a.dtype}") # int32
print(f"Data type of b: {b.dtype}") # float64
print(f"Size of a: {a.size}") # 3
print(f"Size of b: {b.size}") # 6
print(f"Item size of a: {a.itemsize} bytes") # 4 bytes (int32)
print(f"Item size of b: {b.itemsize} bytes") # 8 bytes (float64)
print(f"Total bytes consumed by a: {a.nbytes} bytes") # 12 bytes (3 * 4)
print(f"Total bytes consumed by b: {b.nbytes} bytes") # 48 bytes (6 * 8)

# Accessing/Changing specific elements, rows, columns, etc.

