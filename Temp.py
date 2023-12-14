import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from maze import Maze
import random
import math
import tracemalloc
import time

class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def visualize_environment(self, environment, paths=None, start=None, goal=None):
        self.ax.clear()
        
        # Convert the environment to a numeric data type
        environment = environment.astype(float)
        
        self.ax.imshow(environment, cmap='Greys', origin='upper')

        if paths:
            for path in paths:
                path = np.array(path).T
                self.ax.plot(path[1], path[0], marker='o', color='gray', linestyle='--')

        if start:
            self.ax.plot(start[1], start[0], marker='o', color='red', label='Start')

        if goal:
            self.ax.plot(goal[1], goal[0], marker='o', color='green', label='Goal')

        self.ax.legend()
        plt.pause(0.2)

    def bfs_graph_search(self, environment, start, goal, visualize=True):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        queue = deque([(start, [])])
        paths_explored = []
        visited = set()

        while queue:
            current, path = queue.popleft()

            if current == goal:
                paths_explored.append(path + [current])
                if visualize:
                    visualizer.visualize_environment(environment, paths=paths_explored, start=start_position, goal=goal_position)
                return path + [current]

            if current in visited:
                continue

            visited.add(current)
            paths_explored.append(path + [current])

            if visualize:
                visualizer.visualize_environment(environment, paths=paths_explored, start=start_position, goal=goal_position)

            for dx, dy in directions:
                new_x, new_y = current[0] + dx, current[1] + dy



                # for iox, ioy in zip(ox, oy):
                #     d = math.hypot(iox - x, ioy - y)

                # if d < self.rr:                # Be careful with this line, you can change '<' to '<=' to see the differrences of new_array defined laterr!
                #     self.obmap[ix][iy] = True

                # new_array = np.zeros(np.shape(self.obmap))
                # for i in range(new_array.shape[0]):
                #     for j in range(new_array.shape[1]): 
                #         if self.obmap[i][j]==True:
                #             new_array[i][j]=1   

                # if self.obmap[node.x][node.y]:
                #     return False


                if (
                    0 <= new_x < len(environment)
                    and 0 <= new_y < len(environment[0])
                    and environment[new_x][new_y] == 0
                    and (new_x, new_y) not in visited
                ):
                    queue.append(((new_x, new_y), path + [current]))

        return None

    def move_robot_along_path(self, environment, start, goal, path):
        if path is not None:
            for position in path:
                visualizer.visualize_environment(environment, paths=[path], start=start, goal=goal)
        else:
            print("No path found.")

    def run_search_algorithm(self, environment, start, goal, visualize=True):
        tracemalloc.start()

        start_time = time.time()

        path = self.bfs_graph_search(environment, start, goal, visualize)

        end_time = time.time()
        exec_time = end_time - start_time

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # CAN ADD PATH TO RETURN
        return path, exec_time, peak
    
    def euclidean_distance(self, rows, cols):
        distance = math.sqrt((rows-1 - 0)**2 + (cols-1 - 0)**2)
        return math.floor(distance)

    def calculate_obstacle_density(self, maze_matrix):
        total_cells = len(maze_matrix) * len(maze_matrix[0])
        obstacle_cells = sum(row[1:-1].count(1) for row in maze_matrix[1:-1])
        density_percentage = (obstacle_cells / total_cells) * 100
        return density_percentage


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizing = True
    random.seed() # FOR REPRODUCABILITY 

    if visualizing:
        # SINGLE RUN: VISUALIZATION
        rows = random.randint(15, 40)
        cols = random.randint(15, 40)
        seed = random.randint(1, 1000)
        cutting_rate = random.uniform(0.4, 0.9)
        goal_and_start_spacing = random.randint(round(0.25*visualizer.euclidean_distance(rows, cols)), round(0.85*visualizer.euclidean_distance(rows, cols)))

        # RANDOM MAZE
        # maze = Maze.Maze(rows=rows, cols=cols, space_step=3, seed=seed, remove_lone_blocks=True)
        # environment = maze.generate_maze(rand=cutting_rate) 
        # start_position, goal_position = maze.set_start_and_goal(goal_and_start_spacing)

        # STATIC MAZE (comment above then uncomment this to set static)
        maze = Maze(rows=10, cols=10, space_step=None, seed=45, remove_lone_blocks=False)

        environment = maze.generate_maze(rand=0)
        print(environment)
        start_position, goal_position = maze.set_start_and_goal(5)

        path_graph_search, execution_time, peak_memory = visualizer.run_search_algorithm(environment, start_position, goal_position, visualize=True)
        visualizer.move_robot_along_path(environment, start_position, goal_position, path_graph_search)
        plt.show()
    else:
        # MULTIPLE RUNS: AVERAGE SCORES
        num_runs = 100
        total_time = 0
        total_memory = 0

        for i in range(num_runs):
            rows = random.randint(15, 50)
            cols = random.randint(15, 50)
            seed = random.randint(1, 1000)
            cutting_rate = random.uniform(0.35, 0.85)
            goal_and_start_spacing = random.randint(round(0.25*visualizer.euclidean_distance(rows, cols)), round(0.85*visualizer.euclidean_distance(rows, cols))) 
            # Random range from 25% and 85% of the hypotenuse of the 2 dimensions. Proportional to maze size, beyond Q1 statistically.

            maze = Maze.Maze(rows=rows, cols=cols, space_step=3, seed=seed, remove_lone_blocks=True)
            environment = maze.generate_maze(rand=cutting_rate)
            start_position, goal_position = maze.set_start_and_goal(goal_and_start_spacing) # Set to 5-10 if randomized spacing disrupting output

            density = visualizer.calculate_obstacle_density(maze.matrix)

            # DO NOT UN-COMMENT IF YOUR NUM_RUN IS HIGH. Shows all plots visually for matrices generated
            """            
            fig, ax = plt.subplots()
            ax.imshow(environment, cmap='Greys', origin='upper')
            ax.set_title(f"Run {i + 1}")
            plt.pause(1)  # Pause for a short duration to display the plot
            """

            path, exec_time, peak_memory = visualizer.run_search_algorithm(environment, start_position, goal_position, visualize=False)
            total_time += exec_time
            total_memory += peak_memory
            print(f"rows: {rows}")
            print(f"cols: {cols}")
            print(f"seed: {seed}")
            print(f"cutting_rate: {cutting_rate}")
            print(f"path: {path}")
            print(f"goal_and_start_spacing: {goal_and_start_spacing}")
            print(f"density: {density}")

            # OTHER POTENTIAL PERFORMANCE VALUES: PATH LENGTH RATIO TO START AND GOAL LENGTH, SEARCH SPACE COVERAGE, BRANCHING FACTOR
            # time.sleep(3)

        avg_time = total_time / num_runs
        avg_memory = total_memory / num_runs
        print(f"Total Number of runs: {num_runs}")
        print(f"Average Execution Time: {avg_time} seconds")
        print(f"Average Peak Memory: {avg_memory / (1024 * 1024)} MB")

"""
Iterations: 100        
Average Execution Time: 0.001598355770111084 seconds
Average Peak Memory: 0.0288818359375 MB

Average Execution Time: 0.0038525700569152833 seconds
Average Peak Memory: 0.10086318969726563 MB

Iterations: 1000
Average Execution Time: 0.0020528242588043213 seconds
Average Peak Memory: 0.03667719268798828 MB

Average Execution Time: 0.006160878658294678 seconds
Average Peak Memory: 0.17426569366455077 MB

Iterations: 10000
Average Execution Time: 0.0007655791521072388 seconds
Average Peak Memory: 0.019989265537261963 MB

Average Execution Time: 0.005421704435348511 seconds
Average Peak Memory: 0.13456009302139282 MB
"""


