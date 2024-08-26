from collections import deque
import numpy as np

class BFSPathfinder():
    def __init__(self, start):
        self.start = start

        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    def find_path(self, goal, grid):
        n = len(grid)
        queue = deque([self.start])
        visited = set()
        visited.add(self.start)
        parent = {self.start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                return self.reconstruct_path(parent, goal)

            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                if (self.is_valid(neighbor, grid, n) and neighbor not in visited):
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

        return None  # No path found

    def is_valid(self, position, grid, n):
        row, col = position
        return (0 <= row < n and
                0 <= col < n and
                grid[row][col] == 0)

    def reconstruct_path(self, parent, goal):
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        return path[::-1]  # Return reversed path


# Example usage
grid = np.array([[0, 1, 0, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 0, 0, 1, 0],
                 [1, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0]])

start = (2, 4)  # Starting position (row, col)
goal = (4, 0)   # Goal position (row, col)

pathfinder = BFSPathfinder(start = (2, 4))
path = pathfinder.find_path(goal, grid)

if path:
    print("Path found:", path)
else:
    print("No path found")
