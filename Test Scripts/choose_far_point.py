import numpy as np

def find_farthest_point(binary_array):
    # Get the indices of all points with value 1
    points = np.argwhere(binary_array == 1)
    
    if len(points) <= 1:
        # If there's only one or no point, return the point itself or None
        return points[0] if len(points) == 1 else None
    
    max_distance = 0
    farthest_point = points[0]
    
    # Compare each point to every other point
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Calculate Euclidean distance between points[i] and points[j]
            distance = np.linalg.norm(points[i] - points[j])
            # Update the farthest point if this distance is the largest so far
            if distance > max_distance:
                max_distance = distance
                farthest_point = points[j]
    
    return tuple(farthest_point)

# Example usage:
binary_array = np.array([
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

farthest_point = find_farthest_point(binary_array)
print("Farthest point:", farthest_point)
