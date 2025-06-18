import numpy as np
print(np.__version__)

# Array creation
a = np.array([1, 2, 3], dtype='int32')
print(a)

b = np.array([[9.0, 8.0, 7.0], [6, 5, 4]])
print(b)

# Array properties
print(f"Number of dimensions of a: {a.ndim}")  # 1 dimension
print(f"Number of dimensions of b: {b.ndim}")  # 2 dimensions
print(f"Shape of a: {a.shape}")  # (3,)
print(f"Shape of b: {b.shape}")  # (2, 3)
print(f"Data type of a: {a.dtype}")  # int32
print(f"Data type of b: {b.dtype}")  # float64
print(f"Size of a: {a.size}")  # 3
print(f"Size of b: {b.size}")  # 6
print(f"Item size of a: {a.itemsize} bytes")  # 4 bytes (int32)
print(f"Item size of b: {b.itemsize} bytes")  # 8 bytes (float64)
print(f"Total bytes consumed by a: {a.nbytes} bytes")  # 12 bytes (3 * 4)
print(f"Total bytes consumed by b: {b.nbytes} bytes")  # 48 bytes (6 * 8)

# Accessing/Changing specific elements, rows, columns, etc.

a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
print(f"Element at row 1, column 5: {a[1, 5]}")  # 13
print(f"Get 1st row: {a[0, :]}")  # [1 2 3 4 5 6 7]
print(f"Get 2nd column: {a[:, 2]}")  # [ 3 10]

# Getting a little fancy [startindex:endindex:stepsize]
print(f"Get 0th row, 1st to 5th column, step 2: {a[0, 1:5:2]}")  # [2 4]

a[:, 2] = 20
print(f"a[:, 2] = 20\n{a}")  # Change all values in 2nd column to 20

# 3D array
b = np.array([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

# Get specific element (work outside in)

print(f'Get specific element at b[0, 1, 1]: {b[0, 1, 1]}')  # 4
print(f'Get 2nd row from both 2D arrays: {b[:, 1, :]}')  # [[3 4] [7 8]]

# Replace row from both 2D arrays
b[:, 1, :] = [[9, 10], [11, 12]]
# Replace 2nd row with new values
print(f'Replace row from both 2D arrays:\n{b}')

# Initializing different types of arrays
zeros = np.zeros((2, 3))
print(f"All zeros:\n{zeros}")  # 2x3 array of zeros

zeros = np.zeros((2, 3, 3))
print(f"All zeros (3D):\n{zeros}")  # 2x3x3 array of zeros

np_ones = np.ones((2, 3), dtype='int32')
print(f"All ones:\n{np_ones}")  # 2x3 array of ones

# Any other number
np_full = np.full((2, 3), 5)
print(f"All fives:\n{np_full}")  # 2x3 array filled with 5

# Any other number (full_like)
np_full_like = np.full_like(a, 4)
print(f"Full like a with 4:\n{np_full_like}")  # 2x7 array filled with 4

# Random decimal numbers
np_random = np.random.rand(2, 3)
print(f"Random decimal numbers:\n{np_random}")  # 2x3 array of random floats

# Random samples
np_random_samples = np.random.random_sample(a.shape)
# Random samples with shape of a
print(f"Random samples with shape of a:\n{np_random_samples}")

# Random integers
np_random_ints = np.random.randint(0, 10, size=(2, 3))
# 2x3 array of random integers
print(f"Random integers between 0 and 10:\n{np_random_ints}")

# Identity matrix
identity_matrix = np.identity(3)
print(f"Identity matrix:\n{identity_matrix}")  # 3x3 identity matrix

# Repeat an 1D array
arr = np.array([1, 2, 3])
arr_repeated = np.repeat(arr, 3, axis=0)
print(f"Array repeated 3 times:\n{arr_repeated}")  # Repeat elements

# Repeat a 2D array
arr = np.array([[1, 2, 3]])
arr_repeated = np.repeat(arr, 3, axis=0)  # Repeat rows
print(f"Array repeated 3 times along axis 0:\n{arr_repeated}")  # Repeat rows

arr_repeated = np.repeat(arr, 3, axis=1)  # Repeat columns
# Repeat columns
print(f"Array repeated 3 times along axis 1:\n{arr_repeated}")

# Repeat a 3D array
arr = np.array([[[1, 2], [3, 4]]])
arr_repeated = np.repeat(arr, 3, axis=0)  # Repeat along the first axis
# Repeat along the first axis
print(f"3D array repeated 3 times along axis 0:\n{arr_repeated}")

arr_repeated = np.repeat(arr, 2, axis=1)  # Repeat along the second axis
# Repeat along the second axis
print(f"3D array repeated 2 times along axis 1:\n{arr_repeated}")

arr_repeated = np.repeat(arr, 2, axis=2)  # Repeat along the third axis
# Repeat along the third axis
print(f"3D array repeated 2 times along axis 2:\n{arr_repeated}")

# Creating a 5x5 array with a border of ones and an inner area of zeros and a 9 in the center

'''
[[1 1 1 1 1]
 [1 0 0 0 1]
 [1 0 9 0 1]
 [1 0 0 0 1]
 [1 1 1 1 1]]
'''
output = np.ones((5, 5), dtype='int32')  # Create a 5x5 array of ones
print(f"Output array with ones:\n{output}")  # 5x5 array of ones

z = np.zeros((3, 3), dtype='int32')  # Create a 3x3 array of zeros
print(f"Zero array:\n{z}")  # 3x3 array of zeros

output[1: -1, 1: -1] = z  # Set the inner 3x3 area to zeros
# 5x5 array with inner area zeros
print(f"Output array with inner area set to zeros:\n{output}")

output[2, 2] = 9  # Set the center element to 9
print(f"Output array with center set to 9:\n{output}")  # Final output

# Copying arrays
a = np.array([1, 2, 3])
b = a.copy()  # Create a copy of a
print(f"Original array a: {a}")  # [1 2 3]
print(f"Copied array b: {b}")  # [1 2 3]
b[0] = 100  # Change the first element of b
# a remains unchanged
print(f"After changing b[0] to 100:\nOriginal a: {a}\nCopied b: {b}")

# Mathematical operations
a = np.array([1, 2, 3])

print(f"a + 2: {a + 2}")  # [3 4 5]
print(f"a - 2: {a - 2}")  # [-1  0  1]
print(f"a * 2: {a * 2}")  # [2 4 6]
print(f"a / 2: {a / 2}")  # [0.5 1.  1.5]
print(f"a ** 2: {a ** 2}")  # [1 4 9]
print(f"10 * a: {10 * a}")  # [10 20 30]
print(f"a % 2: {a % 2}")  # [1 0 1] (modulus operation)
print(f"a // 2: {a // 2}")  # [0 1 1] (floor division)
# Trigonometric functions
print(f"sin(a): {np.sin(a)}")  # Sine of each element
print(f"cos(a): {np.cos(a)}")  # Cosine of each element
print(f"tan(a): {np.tan(a)}")  # Tangent of each element
# Logarithmic functions
print(f"log(a): {np.log(a)}")  # Natural logarithm of each element
print(f"log10(a): {np.log10(a)}")  # Base-10 logarithm of each element
print(f"log2(a): {np.log2(a)}")  # Base-2 logarithm of each element
# Exponential function
print(f"exp(a): {np.exp(a)}")  # Exponential of each element
# Square root
print(f"sqrt(a): {np.sqrt(a)}")  # Square root of each element
# Aggregation functions
print(f"Sum of a: {np.sum(a)}")  # Sum of all elements in a
print(f"Mean of a: {np.mean(a)}")  # Mean of all elements in a
print(f"Standard deviation of a: {np.std(a)}")  # Standard deviation of a
print(f"Variance of a: {np.var(a)}")  # Variance of a
print(f"Minimum of a: {np.min(a)}")  # Minimum value in a
print(f"Maximum of a: {np.max(a)}")  # Maximum value in a
print(f"Sum of a (axis=0): {np.sum(a, axis=0)}")  # Sum along axis 0
print(f"Mean of a (axis=0): {np.mean(a, axis=0)}")  # Mean along axis 0
# Std dev along axis 0
print(f"Standard deviation of a (axis=0): {np.std(a, axis=0)}")
print(f"Variance of a (axis=0): {np.var(a, axis=0)}")  # Variance along axis 0
print(f"Minimum of a (axis=0): {np.min(a, axis=0)}")  # Minimum along axis 0
print(f"Maximum of a (axis=0): {np.max(a, axis=0)}")  # Maximum along axis 0
# Linear algebra operations
# Dot product
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(f"Dot product of a and b:\n{np.dot(a, b)}")  # Dot product of a and b
# Matrix multiplication
# Matrix multiplication of a and b
print(f"Matrix multiplication of a and b:\n{a @ b}")
# Transpose of a matrix
print(f"Transpose of a:\n{a.T}")  # Transpose of a
# Determinant of a matrix
print(f"Determinant of a:\n{np.linalg.det(a)}")  # Determinant of a
# Inverse of a matrix
try:
    print(f"Inverse of a:\n{np.linalg.inv(a)}")  # Inverse of a
except np.linalg.LinAlgError as e:
    print(f"Error inverting matrix a: {e}")
# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(a)
print(f"Eigenvalues of a: {eigenvalues}")  # Eigenvalues of a
print(f"Eigenvectors of a:\n{eigenvectors}")  # Eigenvectors of a
# Reshaping arrays
# Reshape a 1D array to a 2D array
a = np.array([1, 2, 3, 4, 5, 6])
reshaped_a = a.reshape((2, 3))  # Reshape to 2 rows and 3 columns
print(f"Reshaped a to 2D array:\n{reshaped_a}")  # Reshaped array
# Flatten a 2D array to a 1D array
flattened_a = reshaped_a.flatten()  # Flatten the 2D array to 1D
print(f"Flattened a to 1D array:\n{flattened_a}")  # Flattened array
# Transpose a 2D array
transposed = reshaped_a.T  # Transpose the 2D array
print(f"Transposed a:\n{transposed}")  # Transposed array
# Stacking arrays
# Stack two arrays vertically
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])
stacked_vertical = np.vstack((a, b))  # Stack vertically
print(f"Stacked a and b vertically:\n{stacked_vertical}")  # Stacked array
# Stack two arrays horizontally
stacked_horizontal = np.hstack((a, b))  # Stack horizontally
print(f"Stacked a and b horizontally:\n{stacked_horizontal}")  # Stacked array
# Concatenate arrays along a specific axis
concatenated = np.concatenate((a, b), axis=0)  # Concatenate along axis 0
# Concatenated array
print(f"Concatenated a and b along axis 0:\n{concatenated}")
# Concatenate along axis 1
concatenated_axis1 = np.concatenate((a, b), axis=1)  # Concatenate along axis 1
# Concatenated array
print(f"Concatenated a and b along axis 1:\n{concatenated_axis1}")
# Splitting arrays
# Split an array into two equal parts
split_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
split_arrays = np.array_split(split_a, 2)  # Split into 2 parts
print(f"Split array into 2 parts:\n{split_arrays}")  # Split arrays
# Split an array into 3 equal parts
split_arrays_3 = np.array_split(split_a, 3)  # Split into 3 parts
print(f"Split array into 3 parts:\n{split_arrays_3}")  # Split arrays
# Split an array into 3 parts along axis 1
split_arrays_axis1 = np.array_split(split_a, 3, axis=1)  # Split along axis 1
# Split arrays
print(f"Split array into 3 parts along axis 1:\n{split_arrays_axis1}")
# Boolean indexing
# Create a 1D array
a = np.array([1, 2, 3, 4, 5, 6])
# Create a boolean mask
mask = a > 3  # Mask for elements greater than 3
# [False False False  True  True  True]
print(f"Boolean mask for elements greater than 3: {mask}")
# Use the mask to filter the array
filtered_a = a[mask]  # Filtered array
print(f"Filtered array with elements greater than 3: {filtered_a}")  # [4 5 6]
# Create a 2D array
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Create a boolean mask for elements greater than 5
mask_2d = b > 5  # Mask for elements greater than 5
print(f"Boolean mask for elements greater than 5:\n{mask_2d}")
# [[False False False]
#  [False False  True]
#  [ True  True  True]]
# Use the mask to filter the 2D array
filtered_b = b[mask_2d]  # Filtered 2D array
# [6 7 8 9]
print(f"Filtered 2D array with elements greater than 5:\n{filtered_b}")
# Create a boolean mask for even elements
mask_even = b % 2 == 0  # Mask for even elements
print(f"Boolean mask for even elements:\n{mask_even}")
# [[False  True False]
#  [ True False  True]
#  [False  True False]]
# Use the mask to filter the 2D array for even elements
filtered_even_b = b[mask_even]  # Filtered 2D array for even elements
print(f"Filtered 2D array with even elements:\n{filtered_even_b}")  # [2 4 6 8]
# Create a boolean mask for odd elements
mask_odd = b % 2 != 0  # Mask for odd elements
print(f"Boolean mask for odd elements:\n{mask_odd}")
# [[ True False  True]
#  [False  True False]
#  [ True False  True]]
# Use the mask to filter the 2D array for odd elements
filtered_odd_b = b[mask_odd]  # Filtered 2D array for odd elements
print(f"Filtered 2D array with odd elements:\n{filtered_odd_b}")  # [1 3 5 7 9]
# Sorting arrays
# Create a 1D array
a = np.array([3, 1, 4, 2, 5])
# Sort the array in ascending order
sorted_a = np.sort(a)  # Sort in ascending order
print(f"Sorted 1D array in ascending order: {sorted_a}")  # [1 2 3 4 5]
# Sort the array in descending order
sorted_a_desc = np.sort(a)[::-1]  # Sort in descending order
print(f"Sorted 1D array in descending order: {sorted_a_desc}")  # [5 4 3 2 1]
# Create a 2D array
b = np.array([[3, 1, 4], [2, 5, 6]])
# Sort the 2D array along axis 0 (columns)
sorted_b_axis0 = np.sort(b, axis=0)  # Sort along columns
print(f"Sorted 2D array along axis 0 (columns):\n{sorted_b_axis0}")
# [[2 1 4]
#  [3 5 6]]
# Sort the 2D array along axis 1 (rows)
sorted_b_axis1 = np.sort(b, axis=1)  # Sort along rows
print(f"Sorted 2D array along axis 1 (rows):\n{sorted_b_axis1}")
# [[1 3 4]
#  [2 5 6]]
# Sort the 2D array in descending order along axis 0
# Sort along columns in descending order
sorted_b_axis0_desc = np.sort(b, axis=0)[::-1]
print(
    f"Sorted 2D array along axis 0 in descending order:\n{sorted_b_axis0_desc}")
# [[3 5 6]
#  [2 1 4]]
# Sort the 2D array in descending order along axis 1
# Sort along rows in descending order
sorted_b_axis1_desc = np.sort(b, axis=1)[:, ::-1]
print(
    f"Sorted 2D array along axis 1 in descending order:\n{sorted_b_axis1_desc}")
# [[4 3 1]
#  [6 5 2]]
# Search and count elements
# Create a 1D array
a = np.array([1, 2, 3, 4, 5, 6])
# Search for an element in the array
element_to_search = 4
index = np.where(a == element_to_search)  # Find index of the element
# (array([3], dtype=int64),)
print(f"Index of element {element_to_search} in a: {index}")
# Create a 2D array
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Search for an element in the 2D array
element_to_search_2d = 5
index_2d = np.where(b == element_to_search_2d)  # Find index of the element
# (array([1]), array([1]))
print(f"Index of element {element_to_search_2d} in b: {index_2d}")
# Count occurrences of an element in the array
element_to_count = 2
count = np.count_nonzero(a == element_to_count)  # Count occurrences
print(f"Count of element {element_to_count} in a: {count}")  # 1
# Count occurrences of an element in the 2D array
# Count occurrences in 2D array
count_2d = np.count_nonzero(b == element_to_count)
print(f"Count of element {element_to_count} in b: {count_2d}")  # 1
# Unique elements in an array
# Create a 1D array with duplicate elements
a = np.array([1, 2, 2, 3, 4, 4, 5])
# Find unique elements in the array
unique_elements = np.unique(a)  # Find unique elements
print(f"Unique elements in a: {unique_elements}")  # [1 2 3 4 5]
# Create a 2D array with duplicate elements
b = np.array([[1, 2, 2], [3, 4, 4], [5, 6, 6]])
# Find unique elements in the 2D array
unique_elements_2d = np.unique(b)  # Find unique elements in 2D array
print(f"Unique elements in b: {unique_elements_2d}")  # [1 2 3 4 5 6]
# Find unique rows in a 2D array
unique_rows = np.unique(b, axis=0)  # Find unique rows in 2D array
print(f"Unique rows in b:\n{unique_rows}")  # Unique rows in 2D array
# [[1 2 2]
#  [3 4 4]
#  [5 6 6]]
# Find unique columns in a 2D array
unique_columns = np.unique(b, axis=1)  # Find unique columns in 2D array
print(f"Unique columns in b:\n{unique_columns}")  # Unique columns in 2D array
# [[1 2]
#  [3 4]
#  [5 6]]
# Set operations
# Create two arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])
# Find the intersection of two arrays
intersection = np.intersect1d(a, b)  # Find common elements
print(f"Intersection of a and b: {intersection}")  # [4 5]
# Find the union of two arrays
union = np.union1d(a, b)  # Find all unique elements
print(f"Union of a and b: {union}")  # [1 2 3 4 5 6 7 8]
# Find the difference between two arrays
difference = np.setdiff1d(a, b)  # Elements in a not in b
print(f"Difference of a and b (a - b): {difference}")  # [1 2 3]
# Find the symmetric difference between two arrays
# Elements in either a or b but not both
symmetric_difference = np.setxor1d(a, b)
# [1 2 3 6 7 8]
print(f"Symmetric difference of a and b: {symmetric_difference}")
# Create a 2D array
c = np.array([[1, 2, 3], [4, 5, 6]])
# Find the intersection of two 2D arrays
d = np.array([[4, 5, 6], [7, 8, 9]])
intersection_2d = np.intersect1d(c, d)  # Find common elements in 2D arrays
print(f"Intersection of c and d: {intersection_2d}")  # [4 5 6]
# Find the union of two 2D arrays
union_2d = np.union1d(c, d)  # Find all unique elements in 2D arrays
print(f"Union of c and d: {union_2d}")  # [1 2 3 4 5 6 7 8 9]
# Find the difference between two 2D arrays
difference_2d = np.setdiff1d(c, d)  # Elements in c not in d
print(f"Difference of c and d (c - d): {difference_2d}")  # [1 2 3]
# Find the symmetric difference between two 2D arrays
# Elements in either c or d but not both
symmetric_difference_2d = np.setxor1d(c, d)
# [1 2 3 7 8 9]
print(f"Symmetric difference of c and d: {symmetric_difference_2d}")
# Masked arrays
# Create a 1D array
a = np.array([1, 2, 3, 4, 5, 6])
# Create a mask for elements greater than 3
mask = a > 3  # Mask for elements greater than 3
# [False False False  True  True  True]
print(f"Mask for elements greater than 3: {mask}")
# Create a masked array using the mask
masked_a = np.ma.masked_array(a, mask=mask)  # Masked array
# [1 2 3 -- -- --]
print(f"Masked array with elements greater than 3 masked:\n{masked_a}")
# Create a 2D array
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Create a mask for elements greater than 5
mask_2d = b > 5  # Mask for elements greater than 5
print(f"Mask for elements greater than 5:\n{mask_2d}")
# [[False False False]
#  [False False  True]
#  [ True  True  True]]
# Create a masked array using the mask
masked_b = np.ma.masked_array(b, mask=mask_2d)  # Masked array
print(f"Masked array with elements greater than 5 masked:\n{masked_b}")
# [[1 2 3]
#  [4 5 --]
#  [-- -- --]]
# Create a mask for even elements
mask_even = b % 2 == 0  # Mask for even elements
print(f"Mask for even elements:\n{mask_even}")
# [[False  True False]
#  [ True False  True]
#  [False  True False]]
# Create a masked array using the mask for even elements
# Masked array for even elements
masked_even_b = np.ma.masked_array(b, mask=mask_even)
print(f"Masked array with even elements masked:\n{masked_even_b}")
# [[1 -- 3]
#  [-- 5 --]
#  [7 -- 9]]
# Create a mask for odd elements
mask_odd = b % 2 != 0  # Mask for odd elements
print(f"Mask for odd elements:\n{mask_odd}")
# [[ True False  True]
#  [False  True False]
#  [ True False  True]]
# Create a masked array using the mask for odd elements
# Masked array for odd elements
masked_odd_b = np.ma.masked_array(b, mask=mask_odd)
print(f"Masked array with odd elements masked:\n{masked_odd_b}")
# [[-- 2 --]
#  [4 -- 6]
#  [-- 8 --]]
# Masked array operations
# Create a masked array
# Masked array with element 3 masked
a = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
print(f"Masked array a:\n{a}")  # [1 2 -- 4 5]
# Perform operations on the masked array
print(f"Sum of masked array a: {a.sum()}")  # Sum ignoring masked elements
print(f"Mean of masked array a: {a.mean()}")  # Mean ignoring masked elements
# Std dev ignoring masked elements
print(f"Standard deviation of masked array a: {a.std()}")
# Variance ignoring masked elements
print(f"Variance of masked array a: {a.var()}")
# Minimum ignoring masked elements
print(f"Minimum of masked array a: {a.min()}")
# Maximum ignoring masked elements
print(f"Maximum of masked array a: {a.max()}")
# Count of non-masked elements
print(f"Count of masked elements in a: {a.count()}")
# Fill masked values with 0
print(f"Masked array a after filling masked values with 0:\n{a.filled(0)}")
# Fill masked values with -1
print(f"Masked array a after filling masked values with -1:\n{a.filled(-1)}")
# Create a 2D masked array
# Masked array with some elements masked
b = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 0, 1], [0, 1, 0]])
print(f"Masked array b:\n{b}")  # [[1 2 --] [4 -- 6]]
# Perform operations on the 2D masked array
print(f"Sum of masked array b: {b.sum()}")  # Sum ignoring masked elements
print(f"Mean of masked array b: {b.mean()}")  # Mean ignoring masked elements
# Std dev ignoring masked elements
print(f"Standard deviation of masked array b: {b.std()}")
# Variance ignoring masked elements
print(f"Variance of masked array b: {b.var()}")
# Minimum ignoring masked elements
print(f"Minimum of masked array b: {b.min()}")
# Maximum ignoring masked elements
print(f"Maximum of masked array b: {b.max()}")
# Count of non-masked elements
print(f"Count of masked elements in b: {b.count()}")
# Fill masked values with 0
print(f"Masked array b after filling masked values with 0:\n{b.filled(0)}")
# Fill masked values with -1
print(f"Masked array b after filling masked values with -1:\n{b.filled(-1)}")
# Masked array indexing
# Create a masked array
# Masked array with element 3 masked
a = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
print(f"Masked array a:\n{a}")  # [1 2 -- 4 5]
# Access masked elements
# Access masked element (will show as --)
print(f"Masked element at index 2: {a[2]}")
print(f"Masked elements in a: {a[a.mask]}")  # Access all masked elements
print(f"Unmasked elements in a: {a[~a.mask]}")  # Access all unmasked elements
# Create a 2D masked array
# Masked array with some elements masked
b = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 0, 1], [0, 1, 0]])
print(f"Masked array b:\n{b}")  # [[1 2 --] [4 -- 6]]
# Access masked elements in 2D array
# Access masked element (will show as --)
print(f"Masked element at (0, 2): {b[0, 2]}")
print(f"Masked elements in b: {b[b.mask]}")  # Access all masked elements
print(f"Unmasked elements in b: {b[~b.mask]}")  # Access all unmasked elements
# Masked array slicing
# Create a masked array
# Masked array with element 3 masked
a = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
print(f"Masked array a:\n{a}")  # [1 2 -- 4 5]
# Slice the masked array
sliced_a = a[1:4]  # Slice from index 1 to 3
print(f"Sliced masked array a[1:4]:\n{sliced_a}")  # [2 -- 4]
# Access masked elements in the sliced array
# Access masked elements in sliced array
print(f"Masked elements in sliced a: {sliced_a[sliced_a.mask]}")
# Access unmasked elements in sliced array
print(f"Unmasked elements in sliced a: {sliced_a[~sliced_a.mask]}")
# Create a 2D masked array
# Masked array with some elements masked
b = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 0, 1], [0, 1, 0]])
print(f"Masked array b:\n{b}")  # [[1 2 --] [4 -- 6]]
# Slice the 2D masked array
sliced_b = b[0:2, 1:3]  # Slice rows 0 to 1 and columns 1 to 2
print(f"Sliced masked array b[0:2, 1:3]:\n{sliced_b}")  # [[2 --] [-- 6]]
# Access masked elements in the sliced 2D array
# Access masked elements in sliced array
print(f"Masked elements in sliced b: {sliced_b[sliced_b.mask]}")
# Access unmasked elements in sliced array
print(f"Unmasked elements in sliced b: {sliced_b[~sliced_b.mask]}")
# Masked array operations
# Create a masked array
a = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
print(f"Masked array a:\n{a}")  # [1 2 -- 4 5]
# Perform operations on the masked array
print(f"Sum of masked array a: {a.sum()}")  # Sum ignoring masked elements
print(f"Mean of masked array a: {a.mean()}")  # Mean ignoring masked elements
print(f"Standard deviation of masked array a: {a.std()}")  # Std dev ignoring masked elements
print(f"Variance of masked array a: {a.var()}")  # Variance ignoring masked elements
print(f"Minimum of masked array a: {a.min()}")  # Minimum ignoring masked elements
print(f"Maximum of masked array a: {a.max()}")  # Maximum ignoring masked elements
print(f"Count of masked elements in a: {a.count()}")  # Count of non-masked elements
# Fill masked values with 0
print(f"Masked array a after filling masked values with 0:\n{a.filled(0)}")
# Fill masked values with -1
print(f"Masked array a after filling masked values with -1:\n{a.filled(-1)}")
import numpy as np
# Create a 2D masked array
b = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 0, 1], [0, 1, 0]])
print(f"Masked array b:\n{b}")  # [[1 2 --] [4 -- 6]]
# Perform operations on the 2D masked array
print(f"Sum of masked array b: {b.sum()}")  # Sum ignoring masked elements
print(f"Mean of masked array b: {b.mean()}")  # Mean ignoring masked elements
print(f"Standard deviation of masked array b: {b.std()}")  # Std dev ignoring masked elements
print(f"Variance of masked array b: {b.var()}")  # Variance ignoring masked elements
print(f"Minimum of masked array b: {b.min()}")  # Minimum ignoring masked elements
print(f"Maximum of masked array b: {b.max()}")  # Maximum ignoring masked elements
print(f"Count of masked elements in b: {b.count()}")  # Count of non-masked elements
# Fill masked values with 0
print(f"Masked array b after filling masked values with 0:\n{b.filled(0)}")
# Fill masked values with -1
print(f"Masked array b after filling masked values with -1:\n{b.filled(-1)}")
import numpy as np
# Create a 1D masked array
a = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
print(f"Masked array a:\n{a}")  # [1 2 -- 4 5]
# Access masked elements
print(f"Masked element at index 2: {a[2]}")  # Access masked element (will show as --)
print(f"Masked elements in a: {a[a.mask]}")  # Access all masked elements
print(f"Unmasked elements in a: {a[~a.mask]}")  # Access all unmasked elements
# Create a 2D masked array
b = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 0, 1], [0, 1, 0]])
print(f"Masked array b:\n{b}")  # [[1 2 --] [4 -- 6]]
# Access masked elements in 2D array
print(f"Masked element at (0, 2): {b[0, 2]}")  # Access masked element (will show as --)
print(f"Masked elements in b: {b[b.mask]}")  # Access all masked elements
print(f"Unmasked elements in b: {b[~b.mask]}")  # Access all unmasked elements
# Masked array slicing
# Create a masked array
a = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
print(f"Masked array a:\n{a}")  # [1 2 -- 4 5]
# Slice the masked array
sliced_a = a[1:4]  # Slice from index 1 to 3
print(f"Sliced masked array a[1:4]:\n{sliced_a}")  # [2 -- 4]
# Access masked elements in the sliced array
print(f"Masked elements in sliced a: {sliced_a[sliced_a.mask]}")  # Access masked elements
print(f"Unmasked elements in sliced a: {sliced_a[~sliced_a.mask]}")  # Access unmasked elements
# Create a 2D masked array
b = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 0, 1], [0, 1, 0]])
print(f"Masked array b:\n{b}")  # [[1 2 --] [4 -- 6]]
# Slice the 2D masked array
sliced_b = b[0:2, 1:3]  # Slice rows 0 to 1 and columns 1 to 2
print(f"Sliced masked array b[0:2, 1:3]:\n{sliced_b}")  # [[2 --] [-- 6]]
# Access masked elements in the sliced 2D array
print(f"Masked elements in sliced b: {sliced_b[sliced_b.mask]}")  # Access masked elements
print(f"Unmasked elements in sliced b: {sliced_b[~sliced_b.mask]}")  # Access unmasked elements
import numpy as np
# Create a 1D masked array
a = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
print(f"Masked array a:\n{a}")  # [1 2 -- 4 5]
# Perform operations on the masked array
print(f"Sum of masked array a: {a.sum()}")  # Sum ignoring masked elements
print(f"Mean of masked array a: {a.mean()}")  # Mean ignoring masked elements
print(f"Standard deviation of masked array a: {a.std()}")  # Std dev ignoring masked elements
print(f"Variance of masked array a: {a.var()}")  # Variance ignoring masked elements
print(f"Minimum of masked array a: {a.min()}")  # Minimum ignoring masked elements
print(f"Maximum of masked array a: {a.max()}")  # Maximum ignoring masked elements
print(f"Count of masked elements in a: {a.count()}")  # Count of non-masked elements
# Fill masked values with 0
print(f"Masked array a after filling masked values with 0:\n{a.filled(0)}")
# Fill masked values with -1
print(f"Masked array a after filling masked values with -1:\n{a.filled(-1)}")
import numpy as np
# Create a 2D masked array
b = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[0, 0, 1], [0, 1, 0]])
print(f"Masked array b:\n{b}")  # [[1 2 --] [4 -- 6]]
# Perform operations on the 2D masked array
print(f"Sum of masked array b: {b.sum()}")  # Sum ignoring masked elements
print(f"Mean of masked array b: {b.mean()}")  # Mean ignoring masked elements
print(f"Standard deviation of masked array b: {b.std()}")  # Std dev ignoring masked elements
