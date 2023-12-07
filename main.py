import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import Maze
import random
import math
import tracemalloc
import time

class Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
    """
TO-DO:
    VISUALIZING FUNCTION
    SEARCH ALGORITHM AND PATH TRAVERSAL CODE 
    NODE CLASS (with cost and stuff)
    """

    def run_search_algorithm(self, environment, start, goal, visualize=True):
        tracemalloc.start()

        start_time = time.time()

        #path = self.bfs_graph_search(environment, start, goal, visualize)

        end_time = time.time()
        exec_time = end_time - start_time

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return path, exec_time, peak
    
    def euclidean_distance(self, rows, cols):
        distance = math.sqrt((rows-1 - 0)**2 + (cols-1 - 0)**2)
        return math.floor(distance)

    def calculate_obstacle_density(self, maze_matrix):
        total_cells = len(maze_matrix) * len(maze_matrix[0])
        obstacle_cells = sum(row.count(1) for row in maze_matrix)
        density_percentage = (obstacle_cells / total_cells) * 100
        return density_percentage


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizing = False

    # SINGLE RUN: VISUALIZATION
    if visualizing:
        rows = 35
        cols = 35
        seed = 20
        cutting_rate = 0.5
        goal_and_start_spacing = 15

        maze = Maze.Maze(rows=rows, cols=cols, space_step=3, seed=seed, remove_lone_blocks=True)
        environment = maze.generate_maze(rand=cutting_rate)
        # environment = maze.predefined_maze()
        start_position, goal_position = maze.set_start_and_goal(goal_and_start_spacing)

        path_graph_search, execution_time, peak_memory = visualizer.run_search_algorithm(environment, start_position, goal_position, visualize=True)
        """
        ENTER VISUALING FUNCTION CALLING HERE
        """
        plt.show()
    else:
        # MULTIPLE RUNS: AVERAGE SCORES
        num_runs = 100
        total_time = 0
        total_memory = 0

        for i in range(num_runs):
            rows = random.randint(15, 40)
            cols = random.randint(15, 40)
            seed = random.randint(1, 1000)
            cutting_rate = random.uniform(0.35, 0.75)
            goal_and_start_spacing = random.randint(1, visualizer.euclidean_distance(rows, cols))

            # Move the environment instantiation inside the loop
            maze = Maze.Maze(rows=rows, cols=cols, space_step=3, seed=seed, remove_lone_blocks=True)
            environment = maze.generate_maze(rand=cutting_rate)
            start_position, goal_position = maze.set_start_and_goal(goal_and_start_spacing)

            # Calculate obstacle density for each run
            density = visualizer.calculate_obstacle_density(maze.matrix)

            i, exec_time, peak_memory = visualizer.run_search_algorithm(environment, start_position, goal_position, visualize=False)
            total_time += exec_time
            total_memory += peak_memory
            print(f"rows: {rows}")
            print(f"cols: {cols}")
            print(f"seed: {seed}")
            print(f"cutting_rate: {cutting_rate}")
            print(f"goal_and_start_spacing: {goal_and_start_spacing}")
            print(f"density: {density}")
            print(f"time and memory: {exec_time}, {peak_memory / (1024 * 1024)}")


        avg_time = total_time / num_runs
        avg_memory = total_memory / num_runs
        print(f"Total Number of runs: {num_runs}")
        print(f"Average Execution Time: {avg_time} seconds")
        print(f"Average Peak Memory: {avg_memory / (1024 * 1024)} MB")


"""
Iterations: 100        
Run 1: 
Average Execution Time: 0.001598355770111084 seconds
Average Peak Memory: 0.0288818359375 MB

Run 2: 
Average Execution Time: 0.0038525700569152833 seconds
Average Peak Memory: 0.10086318969726563 MB

Iterations: 1000
Run 1:
Average Execution Time: 0.0020528242588043213 seconds
Average Peak Memory: 0.03667719268798828 MB

Run 2:
Average Execution Time: 0.006160878658294678 seconds
Average Peak Memory: 0.17426569366455077 MB

Iterations: 10000
Run 1:
Average Execution Time: 0.0007655791521072388 seconds
Average Peak Memory: 0.019989265537261963 MB

Run 2:
Average Execution Time: 0.005421704435348511 seconds
Average Peak Memory: 0.13456009302139282 MB
"""