image_path = 'Scripts\\test_image.png'


import cv2
import numpy as np
import matplotlib.pyplot as plt
from array_downscale import downscale_binary_array

# Load the image
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection to identify edges
edges = cv2.Canny(blurred, 50, 150)

# Use morphological operations to enhance the paths
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a binary matrix with the same size as the input image
binary_matrix = np.zeros_like(gray)

# Fill the binary matrix with 1s where the path is detected
for contour in contours:
    cv2.drawContours(binary_matrix, [contour], -1, 255, thickness=cv2.FILLED)

# Convert binary matrix to 0s and 1s
binary_matrix = (binary_matrix == 255).astype(int)


print(binary_matrix.shape)

binary_matrix = downscale_binary_array(binary_matrix, 3)
binary_matrix = downscale_binary_array(binary_matrix, 3)
binary_matrix = downscale_binary_array(binary_matrix, 2)

print(binary_matrix.shape)

# # Print the binary matrix
# for row in binary_matrix:
#     print(''.join(map(str, row)))

# Display the images
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Edges')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Detected Paths as Binary Matrix')
plt.imshow(binary_matrix, cmap='gray')
plt.axis('off')

plt.show()



# 1, 1, 1
# 0, 1, 0 -> 1
# 1, 0, 1

# 0, 1, 0
# 1, 0, 1 - > 0
# 0, 0, 0