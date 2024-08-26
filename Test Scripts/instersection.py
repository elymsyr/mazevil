import numpy as np

# Example NumPy arrays
array1 = np.array([[1, 0], [0, 1], [-1, -1]])
array2 = np.array([[0, 1], [-1, 0], [1, 0], [0, 0]])

# Find the intersection of the two arrays
intersection = np.intersect1d(array1.view([('', array1.dtype)] * array1.shape[1]), 
                              array2.view([('', array2.dtype)] * array2.shape[1]))

# Reshape the intersection back to the original 2D shape
intersection_2d = intersection.view(array1.dtype).reshape(-1, array1.shape[1])

print(intersection_2d)

import numpy as np

# Example NumPy array
array = np.array([[0, 1], [-1, 0], [1, 0], [0, 0]])

# The tuple to check
tuple_to_check = (1, 0)

# Check if the tuple exists in the array
exists = np.any(np.all(array == tuple_to_check, axis=1))

print(exists)  # Output: True
