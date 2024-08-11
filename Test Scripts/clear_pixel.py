import numpy as np
from scipy.ndimage import convolve

# Define the binary array (example)
binary_array = np.array([
    [0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0]
])

# Define the kernel to check the 8 neighbors
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

# Count the number of neighbors that are 1 for each pixel
neighbor_count = convolve(binary_array, kernel, mode='constant', cval=0)

# Apply the condition: Set pixel to 0 if it has less than 3 neighbors that are 1
result_array = np.where(neighbor_count >= 3, binary_array, 0)

print("Original Array:")
print(binary_array)

print("Modified Array:")
print(result_array)