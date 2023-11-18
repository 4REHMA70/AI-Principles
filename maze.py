import random
#
class Maze:
    def __init__(self, rows, cols, seed=None):
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.matrix = [[1] * cols for _ in range(rows)]  # Initialize with walls. For each col in row, turn it into 1.

    def generate_maze(self):
        random.seed(self.seed) # Setting seed
        #print(self.matrix)

        # Initialize starting point and stack for backtracking
        stack = [(1, 1)]  # Starting point
        self.matrix[1][1] = 0  # Set starting point as path. First col, first row is 0

        while stack: #Same logic as search algorithms
            current_cell = stack[-1]
            neighbors = self.get_unvisited_neighbors(current_cell[0], current_cell[1])

            if neighbors:
                next_cell = random.choice(neighbors)
                x, y = next_cell
                nx, ny = current_cell

                # Carve a path
                self.matrix[(x + nx) // 2][(y + ny) // 2] = 0  # Set the cell between current and next as path
                self.matrix[x][y] = 0  # Set the next cell as path

                stack.append(next_cell)
            else:
                stack.pop()  # Backtrack

    def get_unvisited_neighbors(self, x, y):
        neighbors = [(x + dx, y + dy) for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]]
        neighbors = [(nx, ny) for nx, ny in neighbors if 0 < nx < self.rows - 1 and 0 < ny < self.cols - 1 and self.matrix[nx][ny]]
        return [neighbor for neighbor in neighbors if self.matrix[(x + neighbor[0]) // 2][(y + neighbor[1]) // 2]]

    def print_maze(self):
        for row in self.matrix:
            print(' '.join(map(str, row)))
  

    def predefined_maze(self):
        maze_text = [
            "1111111111111111111111111111111",
            "0100000000000000010000000000010",
            "1000100010100010100100010100010",
            "1000100010100010100100010100010",
            "1000100010100010100100010100010",
            "1000000000000000000000000000010",
            "1000100010100010100100010100010",
            "1000100010100010100100010100010",
            "1000000000100000000000100000001",
            "1111101010101010101010101111111",
            "1111101010101010101010101111111",
            "1111101000000000100000001111111",
            "1111101010101010101010101111111",
            "1111101010010100010010101111111",
            "0000000010100100010100100000000",
            "1111101010010100010010101111111",
            "1111101010101010101010101111111",
            "1111101000000000100000001111111",
            "1111101010101010101010101111111",
            "1111101010101010101010101111111",
            "1000000000000000000000000000001",
            "1000100010100010100100010100010",
            "1000100010100010100100010100010",
            "1000000000000010010000000000001",
            "1111111111111111111111111111111",
        ]
        maze= [[int(cell) for cell in row] for row in maze_text]
        self.matrix=maze


# Example Usage
if __name__ == "__main__":
    maze = Maze(rows=9, cols=9, seed=42) #ROW, COL, AND SEED SPECIFIED HERE
    maze.generate_maze()
    maze.print_maze()
    maze.predefined_maze()
    maze.print_maze()
