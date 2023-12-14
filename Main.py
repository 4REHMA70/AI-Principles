import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from maze import Maze
import random
import math
import tracemalloc
import time
import ui

class Robot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def visualize(self, environment, paths=None, start=None, goal=None):
        
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

        # ui_display = ui.UserInterface(np.array(environment))
        # ui_display.run()

    def get_next_steps(self, current, environment, visited, directions, step, radius):
        next_steps = []
        
        for dx, dy in directions:
            last = None
            for i in range(1, step+1):
                new_x, new_y = current[0] + i*dx, current[1] + i*dy
                
                if (new_x, new_y) in visited:
                    continue
                elif self.is_valid(new_x, new_y, environment, radius):
                    visited.add(last)
                    last = (new_x, new_y) 
                else:
                    break

            if last:
                next_steps.append(last)
        
        return next_steps

    def breadth_first_search(self, environment, start, goal, visualize=True, radius=1, step=4):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        queue = deque([(start, [])])
        paths_explored = []
        visited = set()

        while queue:
            current, path = queue.popleft()
            goal_x, goal_y = goal
            current_x, current_y = current
            distance = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)

            if current == goal or distance <= radius:
                paths_explored.append(path + [current])
                if visualize:
                    self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                return path + [current], visited

            if current in visited:
                continue

            visited.add(current)
            paths_explored.append(path + [current])

            if visualize:
                self.visualize(environment, paths=paths_explored, start=start, goal=goal)

            """
            # Can test the step-code over here. Note: Step of the length of the maze is dangerous as it traverses in only that 
            step, consequently sacrificing exploration. 66% solutions missed 
            """

            for next_step in self.get_next_steps(current=current, environment=environment, visited=visited, directions=directions, step=step, radius=radius):
                queue.append((next_step, path + [current])) # Appending only last to queue 
                
        return None

    def run_search_algorithm(self, environment, start, goal, visualize=True):
        tracemalloc.start()

        start_time = time.time()

        result = self.breadth_first_search(environment, start, goal, visualize)
        if result:
            path, visited = result
        else:
            path, visited = None, None

        end_time = time.time()
        exec_time = end_time - start_time

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # CAN ADD PATH TO RETURN
        return path, exec_time, current, peak, visited
    
    def euclidean_distance(self, rows, cols):
        distance = math.sqrt((rows-1 - 0)**2 + (cols-1 - 0)**2)
        return math.floor(distance)

    def calculate_obstacle_density(self, maze_matrix):
        total_cells = len(maze_matrix) * len(maze_matrix[0])
        obstacle_cells = sum(row[1:-1].count(1) for row in maze_matrix[1:-1])
        density_percentage = (obstacle_cells/total_cells) * 100
        return density_percentage

    def single_run(self, rows, cols, seed, cutting_rate, goal_and_start_spacing):
        maze = Maze(rows=rows, cols=cols, space_step=3, seed=seed, remove_lone_blocks=True)
        environment = maze.generate_maze(rand=cutting_rate)
        start, goal = maze.set_start_and_goal(goal_and_start_spacing)
        """
        # Test to prove that the radius checking for goal works 
        environment=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        start, goal = (0,0), (3,3)
        """
        path, execution_time, current, peak_memory, visited = self.run_search_algorithm(environment, start, goal, visualize=True)
        if path is not None:
            for _ in path:
                self.visualize(environment, paths=[path], start=start, goal=goal)
        else:
            print("No solution found!")

    def multiple_runs(self, num_runs):
        total_time = 0
        total_memory = 0
        solutions_count = 0

        for i in range(num_runs):
            rows, cols, seed, cutting_rate, goal_and_start_spacing = self.generate_random_parameters()
            maze = Maze(rows=rows, cols=cols, space_step=3, seed=seed, remove_lone_blocks=True)
            environment = maze.generate_maze(rand=cutting_rate)
            start, goal = maze.set_start_and_goal(goal_and_start_spacing)
            
            # DO NOT UN-COMMENT IF YOUR NUM_RUN IS HIGH. Shows all plots visually for matrices generated
            """
            fig, ax = plt.subplots()
            ax.imshow(environment, cmap='Greys', origin='upper')
            ax.set_title(f"Run {i + 1}")
            plt.pause(1)  # Pause for a short duration to display the plot
            """

            density = self.calculate_obstacle_density(maze.matrix)
            path, exec_time, current, peak_memory, visited = self.run_search_algorithm(environment, start, goal, visualize=False)

            # Because path and visited may be None when radius too big to explore
            if path is not None and visited is not None:
                path_to_spacing_ratio = len(path)/goal_and_start_spacing # The smaller, the better (generally)
                search_space_coverage = len(visited)/(rows*cols)
                solutions_count += 1            
            else:
                path_to_spacing_ratio = None # For when there's no path due to radius size
                search_space_coverage = 0

            total_time += exec_time
            total_memory += peak_memory
            array = [rows, cols, seed, cutting_rate, path, goal_and_start_spacing, density, path_to_spacing_ratio, search_space_coverage*100]
            for label, value in zip(["rows", "cols", "seed", "cutting_rate", "path", "goal_and_start_spacing", "density", "path_to_spacing_ratio", "search_space_coverage (%)"], array):
                print(f"{label}: {value}")
            print()
            # OTHER POTENTIAL PERFORMANCE VALUES: BRANCHING FACTOR

        avg_time = total_time/num_runs
        avg_memory = total_memory/num_runs
        print()
        print(f"Total Number of runs: {num_runs}")
        print(f"Average Execution Time: {avg_time} seconds")
        print(f"Average Peak Memory: {avg_memory / (1024 * 1024)} MB")
        print(f"Total Solutions: {solutions_count}")

    def generate_random_parameters(self):
        rows = random.randint(15, 40)
        cols = random.randint(15, 40)
        seed = random.randint(1, 1000)
        cutting_rate = random.uniform(0.4, 0.85)
        goal_and_start_spacing = random.randint(round(0.25 * self.euclidean_distance(rows, cols)), round(0.85 * self.euclidean_distance(rows, cols)))
        return rows, cols, seed, cutting_rate, goal_and_start_spacing
    
    def is_valid(self, new_x, new_y, environment, radius):
        radius -= 1
        for i in range(max(0, new_x-radius), min(len(environment), new_x+radius+1)):
            for j in range(max(0, new_y-radius), min(len(environment[0]), new_y+radius+1)):
                if environment[i][j] == 1:
                    return False 
        return (
            0 <= new_x < len(environment) 
            and 0 <= new_y < len(environment[0])
            and environment[new_x][new_y] == 0
        )

if __name__ == "__main__":
    robot = Robot()
    visualizing = True 
    random.seed()  # FOR REPRODUCIBILITY!

    # Single Run: Visualization
    if visualizing:
        # Static Parameters (comment and uncomment to select respective options)
        # rows, cols, seed, cutting_rate, goal_and_start_spacing = 10, 10, 3, 0.5, 6

        # Random Parameters
        rows, cols, seed, cutting_rate, goal_and_start_spacing = robot.generate_random_parameters()
        
        robot.single_run(rows, cols, seed, cutting_rate, goal_and_start_spacing)
    else:
        num_runs = 100
        # Multiple Runs: Average Scores
        robot.multiple_runs(num_runs)