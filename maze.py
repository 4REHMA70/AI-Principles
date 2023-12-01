import random
import numpy as np
import math
#
class Maze:
    def __init__(self, rows, cols, space=None, seed=None):
        self.rows = rows
        self.cols = cols
        self.space = space
        self.seed = seed
        self.matrix = [[1] * cols for _ in range(rows)]  # Initialize with walls. For each col in row, turn it into 1.

    def generate_maze(self):
        random.seed(self.seed)  # Setting seed

        # Initialize starting point and stack for backtracking
        stack = [(1, 1)]  # Starting point
        self.matrix[1][1] = 0  # Set starting point as path. First col, first row is 0

        while stack:  # Same logic as search algorithms. While stack isn't empty:
            current_cell = stack[-1]  # Set current cell
            neighbors = self.get_unvisited_neighbors(current_cell[0], current_cell[1])  # Sets the unvisited neighbors

            if neighbors:
                next_cell = random.choice(neighbors)  # If neighbors are there, randomly choose the next cell to explore
                x, y = next_cell
                nx, ny = current_cell

                # Carve a path
                self.matrix[(x + nx) // 2][(y + ny) // 2] = 0  # Set the cell between current and next as path
                self.matrix[x][y] = 0  # Set the next cell as path

                # Introduce randomness to cut across walls
                random_neighbors = self.get_unvisited_neighbors(x, y)
                if random_neighbors and random.random() < 0.7:  # Adjust the probability as needed
                    random_neighbor = random.choice(random_neighbors)
                    rx, ry = random_neighbor
                    self.matrix[(x + rx) // 2][(y + ry) // 2] = 0  # Set a random cell between current and random neighbor as path

                stack.append(next_cell)
            else:
                stack.pop()  # Backtrack
        self.spacing()
        return np.array(self.matrix)

    def spacing(self):
        if self.space:
            for i in range(self.rows):
                for j in range(self.space, self.cols - 1, self.space + 1):
                    if self.matrix[i][j] == 0:
                        self.matrix[i][j - 1] = 0  

    
    def get_unvisited_neighbors(self, x, y): 
        neighbors = [(x + dx, y + dy) for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]] # Defining potential neighbors, each 2 steps away, right, left, up, down. Adding dx and dy to x and y
        neighbors = [(nx, ny) for nx, ny in neighbors if 0 < nx < self.rows - 1 and 0 < ny < self.cols - 1 and self.matrix[nx][ny]] # For valid neighbors, get coords
        return [neighbor for neighbor in neighbors if self.matrix[(x + neighbor[0]) // 2][(y + neighbor[1]) // 2]] # Setting neighbors that have an unvisited cell in between (neighbors with a '1' in the middle indicate an unvisited cell)

    def print_maze(self):
        for row in self.matrix:
            print(' '.join(map(str, row))) # Printing each row in matrix
  

    def predefined_maze(self):
        maze = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], 
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], 
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], 
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], 
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], 
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], 
                [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], 
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
                [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], 
                [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], 
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], 
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], 
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        self.matrix=maze

    def set_start_and_goal(self, min_distance=5):
            free_spaces = [(i, j) for i in range(self.rows) for j in range(self.cols) if self.matrix[i][j] == 0]

            if not free_spaces:
                raise ValueError("No free spaces in the maze.")
            
            while True:
                start, goal = random.sample(free_spaces, 2)
                if math.dist(start,goal) >= min_distance: # If euclidean dist greater than or equal to min_distance. Else repeat random start,goal instantiation
                    return start, goal

if __name__ == "__main__":
    maze = Maze(rows=6, cols=6, space=2, seed=42) # Row, Col, and Seed specified here!
    maze.generate_maze()
    maze.spacing()
    maze.print_maze()
    #maze.predefined_maze()
    maze.set_start_and_goal()
    #maze.print_maze()
