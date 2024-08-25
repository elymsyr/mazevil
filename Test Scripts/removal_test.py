import cv2
import numpy as np

def remove_isolated_black_points(image, kernel_size=3):
    """
    Function to detect isolated black points in a binary image and turn them white.

    Parameters:
        image (numpy.ndarray): Input binary image (0 or 255).
        kernel_size (int): Size of the kernel for dilation and erosion.

    Returns:
        numpy.ndarray: Processed image with isolated black points turned white.
    """
    # Define the kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilation
    dilated = cv2.dilate(image, kernel, iterations=1)

    # Subtract the dilated image from the original to get isolated points
    isolated_black_points = cv2.subtract(dilated, image)

    # Turn isolated black points white
    result = np.where(isolated_black_points == 1, 1, image)

    return result

# Example usage:
if __name__ == "__main__":
    # Define a binary image waith blocks of 1s and some isolated 0s
    example_image = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    # Add some isolated black points
    example_image[2, 7] = 0
    example_image[3, 7] = 0
    example_image[4, 7] = 0

    # Ensure the image is binary (0 or 255)
    example_image = np.where(example_image > 0, 255, 0).astype(np.uint8)

    # Process the image
    processed_image = remove_isolated_black_points(example_image, kernel_size=3)

    # Display the result
    print("Original Image:")
    print(example_image)
    print("\nProcessed Image:")
    print(processed_image)

    # Save or display the result
