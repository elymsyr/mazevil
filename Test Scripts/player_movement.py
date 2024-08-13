import numpy as np

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to check if a position maintains the required distance from all objects
def is_valid_position(pos, object_positions, min_distance):
    for obj in object_positions:
        if euclidean_distance(pos, obj) < min_distance:
            return False
    return True

# Function to find the optimal direction considering object weights

