import random
import numpy as np
import math
from config import *
#
class Maze:
    def __init__(self, rows, cols, seed=None, lone_blocks_rate=None):
        self.rows = rows
        self.cols = cols
        self.space_step = SPACE_STEP
        self.seed = seed
        self.matrix = [[1] * cols for _ in range(rows)]  # Initialize with walls. For each col in row, turn it into 1
        self.lone_blocks_rate = lone_blocks_rate 

    def generate_maze(self, cutting_rate):
        random.seed(self.seed)  # Setting seed

        # Initialize starting point and stack for backtracking
        stack = [(1, 1)]  # Starting point
        self.matrix[1][1] = 0  # Set starting point as path. First col, first row is 0

        while stack:  # Same logic as search algorithms. While stack isn't empty:
            current_cell = stack[-1]  # Set current cell. FILO (First In Last Out) method from DFS
            neighbors = self.get_unvisited_neighbors(current_cell)  # Sets the unvisited neighbors

            if neighbors:
                next_cell = random.choice(neighbors)  # If neighbors are there, randomly choose the next cell to explore

                # Carve a path
                self.matrix[(next_cell[0] + current_cell[0]) // 2][(next_cell[1] + current_cell[1]) // 2] = 0  # Set cell between current and next as path
                self.matrix[next_cell[0]][next_cell[1]] = 0  # Set next cell as path

                # Get random neighbors
                random_neighbors = self.get_unvisited_neighbors(next_cell)

                if random_neighbors and random.random() < cutting_rate:  # Adjust rate/probability of cutting as needed. 
                    # Only proceed with path generation under certain probability
                    random_neighbor = random.choice(random_neighbors)
                    self.matrix[(next_cell[0] + random_neighbor[0]) // 2][(next_cell[1] + random_neighbor[1]) // 2] = 0  # Set cell between current and random neighbor as path. 

                stack.append(next_cell)
            else:
                stack.pop()  # Backtrack
        if self.space_step:
            self.spacing()
        if self.lone_blocks_rate > 0:
            self.remove_isolated_blocks()
        return np.array(self.matrix)

    def get_unvisited_neighbors(self, next_cell): 
        neighbors = [(next_cell[0] + dx, next_cell[1] + dy) for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]] # Defining potential neighbors, each 2 steps away, right, left, up, down. Adding dx and dy to next_cell[0] and next_cell[1]
        neighbors = [(current_cell) for current_cell in neighbors if 0 < current_cell[0] < self.rows - 1 and 0 < current_cell[1] < self.cols - 1 and self.matrix[current_cell[0]][current_cell[1]]] # For valid neighbors, get coords
        return [neighbor for neighbor in neighbors if self.matrix[(next_cell[0] + neighbor[0]) // 2][(next_cell[1] + neighbor[1]) // 2]] # Setting neighbors that have an unvisited cell in between (neighbors with a '1' in the middle indicate an unvisited cell)

    def spacing(self): 
        if self.space_step != 0: 
            # Horizontal spacing
            for i in range(self.rows):
                for j in range(self.space_step, self.cols - 1, self.space_step):
                    if self.matrix[i][j] == 0: 
                        self.matrix[i][j - 1] = 0

            # Vertical spacing
            for j in range(self.cols):
                for i in range(self.space_step, self.rows - 1, self.space_step):
                    if self.matrix[i][j] == 0: 
                        self.matrix[i - 1][j] = 0

    def remove_isolated_blocks(self):
        rows, cols = len(self.matrix), len(self.matrix[0])
        
        # Inner function to get neighbors. Different from get_unvisited_neighbors which is tailored for the wall generation 
        def get_neighbors(row, col):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            neighbors = [(row + direction[0], col + direction[1]) for direction in directions if 0 <= row + direction[0] < rows and 0 <= col + direction[1] < cols and self.matrix[row + direction[0]][col + direction[1]] == 1]
            return neighbors
        
        # Identify and remove isolated blocks
        removed = 0
        for row in range(1, rows-1):
            for col in range(1, cols-1):
                if self.matrix[row][col] == 1 and not get_neighbors(row, col) and random.random() < self.lone_blocks_rate:
                    self.matrix[row][col] = 0
                    removed += 1

    def set_start_and_goal(self, min_distance=5):
            free_spaces = [(i, j) for i in range(self.rows) for j in range(self.cols) if self.matrix[i][j] == 0]

            if not free_spaces:
                raise ValueError("No free spaces in the maze.")
            
            while True:
                start, goal = random.sample(free_spaces, 2)
                if math.dist(start,goal) >= min_distance: # If euclidean dist greater than or equal to min_distance. Else repeat random start,goal instantiation
                    return start, goal