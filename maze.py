import random
import numpy as np
import math
#
class Maze:
    def __init__(self, rows, cols, space_step=None, seed=None, remove_lone_blocks=False):
        self.rows = rows
        self.cols = cols
        self.space_step = space_step
        self.seed = seed
        self.matrix = [[1] * cols for _ in range(rows)]  # Initialize with walls. For each col in row, turn it into 1
        self.remove_lone_blocks = remove_lone_blocks

    def generate_maze(self, rand):
        random.seed(self.seed)  # Setting seed

        # Initialize starting point and stack for backtracking
        stack = [(1, 1)]  # Starting point
        self.matrix[1][1] = 0  # Set starting point as path. First col, first row is 0

        while stack:  # Same logic as search algorithms. While stack isn't empty:
            current_cell = stack[-1]  # Set current cell. FILO (First In Last Out) method from DFS
            neighbors = self.get_unvisited_neighbors(current_cell[0], current_cell[1])  # Sets the unvisited neighbors

            if neighbors:
                next_cell = random.choice(neighbors)  # If neighbors are there, randomly choose the next cell to explore
                x, y = next_cell
                nx, ny = current_cell

                # Carve a path
                self.matrix[(x + nx) // 2][(y + ny) // 2] = 0  # Set cell between current and next as path
                self.matrix[x][y] = 0  # Set next cell as path

                # Get random neighbors
                random_neighbors = self.get_unvisited_neighbors(x, y)

                if random_neighbors and random.random() < rand:  # Adjust rate/probability of cutting as needed. 
                    # Only proceed with path generation under certain probability
                    random_neighbor = random.choice(random_neighbors)
                    rx, ry = random_neighbor
                    self.matrix[(x + rx) // 2][(y + ry) // 2] = 0  # Set cell between current and random neighbor as path. 

                stack.append(next_cell)
            else:
                stack.pop()  # Backtrack
        if self.space_step:
            self.spacing()
        if self.remove_lone_blocks:
            self.remove_isolated_blocks()
        return np.array(self.matrix)

                    
    def get_unvisited_neighbors(self, x, y): 
        neighbors = [(x + dx, y + dy) for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]] # Defining potential neighbors, each 2 steps away, right, left, up, down. Adding dx and dy to x and y
        neighbors = [(nx, ny) for nx, ny in neighbors if 0 < nx < self.rows - 1 and 0 < ny < self.cols - 1 and self.matrix[nx][ny]] # For valid neighbors, get coords
        return [neighbor for neighbor in neighbors if self.matrix[(x + neighbor[0]) // 2][(y + neighbor[1]) // 2]] # Setting neighbors that have an unvisited cell in between (neighbors with a '1' in the middle indicate an unvisited cell)

    def spacing(self): 
        if self.space_step != 0: 
            for i in range(self.rows):
                for j in range(self.space_step, self.cols - 1, self.space_step): # From space to the width with space as step rate
                    if self.matrix[i][j] == 0: 
                        self.matrix[i][j - 1] = 0  

    def remove_isolated_blocks(self):
        rows, cols = len(self.matrix), len(self.matrix[0])
        
        # Inner function to get neighbors. Different from get_unvisited_neighbors which is tailored for the wall generation 
        def get_neighbors(r, c):
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            neighbors = [(r + dr, c + dc) for dr, dc in directions if 0 <= r + dr < rows and 0 <= c + dc < cols and self.matrix[r + dr][c + dc] == 1]
            return neighbors
        
        # Identify and remove isolated blocks
        removed = 0
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                if self.matrix[r][c] == 1 and not get_neighbors(r, c):
                    self.matrix[r][c] = 0
                    removed += 1

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
        self.matrix = maze
        self.rows = len(maze)
        self.cols = len(maze[0])
        return np.array(self.matrix)

    def set_start_and_goal(self, min_distance=5):
            free_spaces = [(i, j) for i in range(self.rows) for j in range(self.cols) if self.matrix[i][j] == 0]

            if not free_spaces:
                raise ValueError("No free spaces in the maze.")
            
            while True:
                start, goal = random.sample(free_spaces, 2)
                if math.dist(start,goal) >= min_distance: # If euclidean dist greater than or equal to min_distance. Else repeat random start,goal instantiation
                    return start, goal

# if __name__ == "__main__":
    # maze = Maze(rows=30, cols=30, space_step=3, seed=30, remove_lone_blocks=True) # Row, Col, and Seed specified here!
    # maze.generate_maze(.45)
    # #maze.remove_isolated_blocks()
    # maze.print_maze()
    # #maze.predefined_maze()
    # maze.set_start_and_goal()
    # #maze.print_maze()
