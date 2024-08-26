import numpy as np

def find_closest_non_zero_neighbor(grid, point):
    x, y = point
    if np.array_equal(grid[x, y], [0, 0, 0]):
        # Define the neighbors' relative positions
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        min_distance = float('inf')
        closest_neighbor = None

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            # Check if the neighbor is within the grid bounds
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if not np.array_equal(grid[nx, ny], [0, 0, 0]):
                    distance = np.linalg.norm(np.array([nx, ny]) - np.array([x, y]))
                    if distance < min_distance:
                        min_distance = distance
                        closest_neighbor = (nx, ny)

        return closest_neighbor
    return point  # Return the original point if it's not [0, 0, 0]

# Example usage
grid = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [1, 2, 3], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

point = (1, 0)
closest_neighbor = find_closest_non_zero_neighbor(grid, point)
print(f"The closest non-zero neighbor is at: {closest_neighbor}")
