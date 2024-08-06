import cv2
import numpy as np

def downscale_binary_array(big_array, block_size):
    # Convert the list of lists into a numpy array
    big_array = np.array(big_array)
    
    # Get the shape of the big array
    big_height, big_width = big_array.shape
    
    # Calculate the dimensions of the smaller array
    small_height = big_height // block_size
    small_width = big_width // block_size
    
    # Initialize the smaller array
    small_array = np.zeros((small_height, small_width), dtype=int)
    
    # Aggregate values in each block
    for i in range(small_height):
        for j in range(small_width):
            # Define the block boundaries
            block = big_array[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            
            # Determine the majority value in the block
            majority_value = np.round(np.mean(block)).astype(int)
            
            # Assign the majority value to the corresponding cell in the smaller array
            small_array[i, j] = majority_value
    
    return small_array

def path_detection(rgb, downscale_order = [9,3], lower_bound = np.array([100, 50, 50]), upper_bound = np.array([255, 150, 150])):
    mask = cv2.inRange(rgb, lower_bound, upper_bound)
    binary_array = (mask > 0).astype(np.uint8)
    for downscale in downscale_order:
        binary_array = downscale_binary_array(binary_array, downscale)
    return binary_array