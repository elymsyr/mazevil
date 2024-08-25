from collections import deque
import numpy as np
class BFSPathfinder:
    def __init__(self, grid):
        self.grid = grid
        self.n = len(grid)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    def find_path(self, start, goal):
        queue = deque([start])
        visited = set()
        visited.add(start)
        parent = {start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                return self.reconstruct_path(parent, start, goal)

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if (self.is_valid(neighbor) and neighbor not in visited):
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

        return None  # No path found

    def is_valid(self, position):
        row, col = position
        return (0 <= row < self.n and
                0 <= col < self.n and
                self.grid[row][col] == 0)

    def reconstruct_path(self, parent, start, goal):
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        return path[::-1]  # Return reversed path

# Example usage
grid = np.array([[0, 1, 0, 0, 0],[0, 1, 0, 1, 0],[0, 0, 0, 1, 0],[1, 1, 0, 0, 0],[0, 0, 0, 1, 0]])

start = (0, 0)  # Starting position (row, col)
goal = (4, 4)   # Goal position (row, col)

pathfinder = BFSPathfinder(grid)
path = pathfinder.find_path(start, goal)

if path:
    print("Path found:", path)
else:
    print("No path found")
