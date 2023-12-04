import random
from collections import deque

class Maze:

    def __init__(self, rows, cols, seed=None):
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.matrix = [[1] * cols for _ in range(rows)] # Matrix initialized with 1s, as obstacles. 0s as free spaces later added

    def generate_maze(self):
        random.seed(self.seed)

        stack = [(1, 1)] # Stack to keep track of visited nodes
        self.matrix[1][1] = 0 # Initial start
        """
        a= [[1,1],1,1,1]
        a is an array, holding [1,1], 1, 1, and 1.
        a[0] references first element, [1,1]
        a[0][0] references element's first element, 1
        """

        while stack: # While stack is running
            current_cell = stack[-1] # Last element from cell. FILO
            neighbors = self.get_unvisited_neighbors(current_cell[0], current_cell[1]) # For current cell, x and y coords passed, to get unvisited neighbors

            if neighbors: # If neighbors exist
                next_cell = random.choice(neighbors) # Randomly choose neighbor as next cell
                current_x, current_y = current_cell
                next_x, next_y = next_cell

                self.matrix[(current_x + next_x) // 2][(current_y + next_y) // 2] = 0 # Midpoint between current and next cell coords, set to free
                self.matrix[current_x][current_y] = 0 # Current coords also set free 

                stack.append(next_cell) # Next cell marked as visited
            else:
                stack.pop() # Popping last cell when no unvisited neighbors (prev cell is now current)

    def get_unvisited_neighbors(self, x, y): 
        neighbors = [(x + dx, y + dy) for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]] # For dx dy in list of tuples, add to x and y to get all combinations
        neighbors = [(nx, ny) for nx, ny in neighbors if 0 < nx < self.rows - 1 and 0 < ny < self.cols - 1 and self.matrix[nx][ny]] # Neighbor validation check
        return [neighbor for neighbor in neighbors if self.matrix[(x + neighbor[0]) // 2][(y + neighbor[1]) // 2]] # Return list of valid neighbors

    """
            set current cell from stack
            get list of unvisited neighboring cells 
            choose random neighbor
            carve passage to neighbor 
            push neighbor to stack as new current cell

            if no more neighbors:
            pop current cell from stack
            make this popped cell the new current cell


            OVERALL:
            GET FIRST CELL AS CURRENT CELL
            RANDOMLY CHOOSE NEIGHBOR (IF NOT VISITED)
            CARVE MIDPOINT AS FREE 
            NEIGHBOR NOW CURRENT CELL

            IF ALL NEIGHBOR CELLS VISITED, BACKTRACK TO PREV CELL AND GO AGAIN
    """ 

    #BFS FUNCTION: TAKES IN GRID, START COORD, GOAL COORD
    def bfs(grid, start, goal):
        ROWS, COLS = len(grid), len(grid[0])
        #VISITED NODES SET
        visited = set([start])
        #POTENTIAL NODES QUEUE (FIFO)
        queue = deque([start])

        length = 0
        
        #WHILE QUEUE NOT EMPTY
        while queue:
            for i in range(len(queue)):
                #FIRST IN, FIRST OUT
                r, c = queue.popleft() # [(1,2),(3,4)]
                """
                ! ! ! !
                """
                print(f"Visiting node ({r}, {c})")

                #IF GOAL, RETURN LENGTH
                if (r, c) == goal:
                    return length

                #NEIGHBORS AT RIGHT LEFT UP DOWN
                neighbors = (r+1, c), (r-1, c), (r, c+1), (r, c-1)
                for row, col in neighbors:
                    #IF ROW AND COL OF NEIGHBOR VALID AND NOT IN VISITED, AND IT'S 0 IN GRID
                    if (0 <= row < ROWS and 0 <= col < COLS and (row, col) not in visited and grid[row][col] == 0):
                        queue.append((row, col))
                        visited.add((row, col))
            
            #INCREASE LENGTH (proceed)
            length += 1

if __name__=="main":
    grid = [[0, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0]]


    maze= Maze(rows=1,cols=2)
    maze.bfs(grid, (0,0), (3,3))
