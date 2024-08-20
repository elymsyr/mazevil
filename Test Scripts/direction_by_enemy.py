import numpy as np
import matplotlib.pyplot as plt

# Predefined directions
directions = np.array([
    [0, 1],   # Up
    [1, 1],   # Up-Right
    [1, 0],   # Right
    [1, -1],  # Down-Right
    [0, -1],  # Down
    [-1, -1], # Down-Left
    [-1, 0],  # Left
    [-1, 1],  # Up-Left
], dtype=np.float64)

def optimal_direction(self, enemies, open_directions):
    distances = np.linalg.norm(enemies[2] - self.center, axis=1)
    enemy_positions = enemies[1]
    weights =  1000 / distances
    
    resultant_vector = np.zeros(2)
    for enemy_position, weight in zip(enemy_positions, weights):
        vector_to_player = self.center - enemy_position
        normalized_vector = vector_to_player / np.linalg.norm(vector_to_player)
        weighted_vector = normalized_vector * weight
        resultant_vector += weighted_vector

    # Find the closest direction
    norms = np.linalg.norm(directions - resultant_vector, axis=1)
    best_direction_index = np.argmin(norms)
    best_direction = directions[best_direction_index]
    
    return int(best_direction[0]), int(best_direction[1])

# Example enemy positions and player position
enemy_positions = np.array([[2, 3], [5, 6], [8, 9]])
player_position = np.array([4, 4])

# Calculate weights as the inverse distance between each enemy and the player
distances = np.linalg.norm(enemy_positions - player_position, axis=1)
weights = 1000 / distances
print(weights)
# Calculate the resultant vector
resultant_vector = np.zeros(2)
for enemy_position, weight in zip(enemy_positions, weights):
    vector_to_player = player_position - enemy_position
    normalized_vector = vector_to_player / np.linalg.norm(vector_to_player)
    weighted_vector = normalized_vector * weight
    resultant_vector += weighted_vector

# Find the closest direction
norms = np.linalg.norm(directions - resultant_vector, axis=1)
best_direction_index = np.argmin(norms)
best_direction = directions[best_direction_index]

print("Best direction to move:", best_direction)

# Visualization
plt.figure(figsize=(8, 8))
plt.grid(True)

# Plot player
plt.scatter(*player_position, color='blue', label='Player', s=100)

# Plot enemies
for idx, (enemy_position, weight) in enumerate(zip(enemy_positions, weights)):
    plt.scatter(*enemy_position, color='red', label=f'Enemy {idx+1}', s=100)
    plt.arrow(enemy_position[0], enemy_position[1], 
              player_position[0] - enemy_position[0], 
              player_position[1] - enemy_position[1], 
              head_width=0.2, head_length=0.2, fc='gray', ec='gray', alpha=0.5)
    plt.text(enemy_position[0], enemy_position[1]-0.2, f'W={weight:.2f}', color='red')

# Plot resultant vector
plt.arrow(player_position[0], player_position[1], 
          resultant_vector[0], resultant_vector[1], 
          head_width=0.3, head_length=0.3, fc='green', ec='green', label='Resultant Vector')

# Plot best direction
plt.arrow(player_position[0], player_position[1], 
          best_direction[0], best_direction[1], 
          head_width=0.3, head_length=0.3, fc='purple', ec='purple', label='Best Direction')

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Player Optimal Movement Direction with Inverse Distance-Based Weights')
plt.show()
