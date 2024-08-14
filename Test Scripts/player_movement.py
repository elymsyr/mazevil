import numpy as np
import heapq

def greedy_best_first_search(start, goal, grid, heuristic):
    rows, cols, _ = grid.shape
    open_list = []
    heapq.heappush(open_list, (heuristic(start), start))
    came_from = {}
    came_from[start] = None
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            break
        
        # Get neighbors
        neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        neighbors = [(x, y) for x, y in neighbors if 0 <= x < rows and 0 <= y < cols and not np.array_equal(grid[x, y], [0, 0, 0])]
        
        for next in neighbors:
            if next not in came_from:
                came_from[next] = current
                priority = heuristic(next)
                heapq.heappush(open_list, (priority, next))
    
    # Reconstruct path
    path = []
    step = goal
    while step:
        path.append(step)
        step = came_from.get(step)
    path.reverse()
    
    # Mark the path in the grid
    path_grid = np.copy(grid)
    for (x, y) in path:
        path_grid[x, y] = [255, 0, 0]  # Mark the path with a color (e.g., red)
    
    return path, path_grid

# Example heuristic function
def manhattan_heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Example grid (3D array where last dimension is color)
grid = np.array([
    [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
    [[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255]],
    [[255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
    [[255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 255]],
    [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
])

start = (0, 0)
goal = (4, 4)

path, path_grid = greedy_best_first_search(start, goal, grid, lambda x: manhattan_heuristic(x, goal))
print("Path found:", path)

# Optionally: print or display the path_grid to visualize the result
