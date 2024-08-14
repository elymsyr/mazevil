import heapq
import math
class Node:
    def __init__(self, x, y, cost):
        self.x = x
        self.y = y
        self.cost = cost
    def __lt__(self, other):
        return self.cost < other.cost
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
def greedy_best_first_search(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])
    pq = []
    heapq.heappush(pq, Node(start[0], start[1], 0))
    visited = set()
    visited.add((start[0], start[1]))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while pq:
        current = heapq.heappop(pq)
        if (current.x, current.y) == goal:
            print(f"Goal reached at ({current.x}, {current.y})")
            return
        for d in directions:
            new_x, new_y = current.x + d[0], current.y + d[1]
            if 0 <= new_x < rows and 0 <= new_y < cols and grid[new_x][new_y] == 0 and (new_x, new_y) not in visited:
                cost = euclidean_distance(new_x, new_y, goal[0], goal[1])
                heapq.heappush(pq, Node(new_x, new_y, cost))
                visited.add((new_x, new_y))
    print("Goal not reachable")

# Example grid
grid = [
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 0, 0]

]

start = (0, 0)
goal = (3, 3)
greedy_best_first_search(grid, start, goal)