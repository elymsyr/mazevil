import cv2
import matplotlib.pyplot as plt
import numpy as np
from array_downscale import downscale_binary_array


# Load the image
image_path = 'Scripts\\test_image.png'
image = cv2.imread(image_path)

# ---------------------------------------------------------------------------------------------

# Convert the image from BGR to RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the RGB range
  # Example lower bound in RGB
  # Example upper bound in RGB

# Create the mask
mask = cv2.inRange(rgb, lower_bound, upper_bound)

binary_array = (mask > 0).astype(np.uint8)
print(binary_array.shape)
binary_array = downscale_binary_array(binary_array, 3)
binary_array = downscale_binary_array(binary_array, 3)
binary_array = downscale_binary_array(binary_array, 3)
print(binary_array.shape)

# # Print the binary matrix
# for row in binary_array:
#     print(''.join(map(str, row)))

# Show the results
# cv2.imshow('Original Image', image)
# cv2.imshow('Mask', mask)


plt.figure(figsize=(12, 8))
plt.title(f'Detected Paths - {binary_array.shape}')
plt.imshow(binary_array, cmap='gray')
plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
