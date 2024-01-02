import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from maze import Maze
from queue import PriorityQueue
import random
import math
import tracemalloc
import time
import ui
from config import *

class Robot:
    def __init__(self, algorithm='bfs'):
        self.fig, self.ax = plt.subplots()
        self.algorithm = algorithm
        # self.directions_cost = {(-1, 0): 1, (1, 0): 1, (0, -1): 1, (0, 1): 1}
        self.directions_cost = {(-1, 0): 1, (1, 0): 1, (0, -1): 1, (0, 1): 1, (-1, -1): 1, (-1, 1): 1, (1, -1): 1, (1, 1): 1}
        # Using PriorityQueue later for UCS
        if algorithm == 'ucs':
            self.open_set = PriorityQueue()
        

    def visualize(self, environment, paths=None, start=None, goal=None):
        
        self.ax.clear()  
        environment = environment.astype(float)
                
        self.ax.imshow(environment, cmap='Greys', origin='upper')  
        
        if paths:
            
            for path in paths:
                path = np.array(path)
                self.ax.plot(path[:,1], path[:,0], marker='o', color='grey', linestyle='-.')

        if start:
            self.ax.plot(start[1], start[0], marker='h', color='blue', markersize=10, label='Start')

        if goal:
            self.ax.plot(goal[1], goal[0], marker='h', color='limegreen', markersize=10, label='Goal')
        
        self.ax.legend(loc='upper left', fontsize=4)   
        
        plt.pause(0.2)

        # ui_display = ui.UserInterface(np.array(environment))
        # ui_display.run()

    def depth_first_search(self, environment, start, goal, visualizing, radius=RADIUS, action_step=3):

        stack = [(start, [])]
        paths_explored = []
        visited = set()

        while stack:
            current, path = stack.pop()
            goal_x, goal_y = goal
            current_x, current_y = current
            distance = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)

            if current == goal or distance <= radius:
                paths_explored.append(path + [current])
                if visualizing:
                    self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                return path + [current], visited

            if current in visited:
                continue

            visited.add(current)
            paths_explored.append(path + [current])

            if visualizing:
                self.visualize(environment, paths=paths_explored, start=start, goal=goal)

            for next_actions in self.get_next_actions(current=current, goal=goal, environment=environment, visited=visited, action_step=action_step, radius=radius):
                stack.append((next_actions, path + [current]))

        return None

    def breadth_first_search(self, environment, start, goal, visualizing, radius=RADIUS, action_step=3):

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
                if visualizing:
                    self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                return path + [current], visited

            if current in visited:
                continue

            visited.add(current)
            paths_explored.append(path + [current])

            if visualizing:
                self.visualize(environment, paths=paths_explored, start=start, goal=goal)

            """
            # Can test the action_step-code over here. Note: Step of the length of the maze is dangerous as it traverses in only that 
            action_step, consequently sacrificing exploration. 66% solutions missed 
            """

            for next_actions in self.get_next_actions(current=current, goal=goal, environment=environment, visited=visited, action_step=action_step, radius=radius):
                queue.append((next_actions, path + [current])) # Appending only last values in each direction to queue 
                
        return None

    def a_star_search(self, environment, start, goal, visualizing, radius=RADIUS, action_step=3):

        open_set = [(start, 0, math.sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2), [])]
        closed_set = set()
        paths_explored = []

        while open_set:
            open_set.sort(key=lambda x: x[1] + x[2])  # Sorting by cost + heuristic
            current, cost, _, path = open_set.pop(0)

            if current == goal:
                paths_explored.append(path + [current])
                if visualizing:
                    self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                return path + [current], closed_set

            if current in closed_set:
                continue

            closed_set.add(current)
            paths_explored.append(path + [current])

            if visualizing:
                self.visualize(environment, paths=paths_explored, start=start, goal=goal)

            for next_actions in self.get_next_actions(current=current, goal=goal, environment=environment, visited=closed_set, action_step=action_step, radius=radius):
                new_cost = cost + 1 
                heuristic_value = math.sqrt((goal[0] - current[0]) ** 2 + (goal[1] - current[1]) ** 2)
                open_set.append((next_actions, new_cost, heuristic_value, path + [current]))

        return None

    def run_search_algorithm(self, environment, start, goal, visualizing, action_step):
        tracemalloc.start()

        start_time = time.time()

        if self.algorithm=='bfs':
            result = self.breadth_first_search(environment, start, goal, visualizing, action_step=action_step)
        elif self.algorithm=='dfs':
            result = self.depth_first_search(environment, start, goal, visualizing, action_step=action_step)
        elif self.algorithm=='a_star':
            result = self.a_star_search(environment, start, goal, visualizing, action_step=action_step)

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
    
    def calculate_obstacle_density(self, maze_matrix):
        total_cells = len(maze_matrix) * len(maze_matrix[0])
        obstacle_cells = sum(row[1:-1].count(1) for row in maze_matrix[1:-1])
        density_percentage = (obstacle_cells/total_cells) * 100
        return density_percentage

    def single_run(self, rows, cols, seed, cutting_rate, goal_and_start_spacing, lone_blocks_rate=1, action_step=3):
        maze = Maze(rows=rows, cols=cols, seed=seed, lone_blocks_rate=lone_blocks_rate)
        environment = maze.generate_maze(rand=cutting_rate)
        start, goal = maze.set_start_and_goal(goal_and_start_spacing)
        """
        # Test to prove that the radius checking for goal works 
        environment=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        start, goal = (0,0), (3,3)
        """
        path, execution_time, current, peak_memory, visited = self.run_search_algorithm(environment, start, goal, visualizing=True, action_step=action_step)
        
        if path is not None:
            for _ in path:
                self.visualize(environment, paths=[path], start=start, goal=goal)
            plt.pause(0.5)
            plt.close()

        else:
            print("No solution found!")

    def multiple_runs(self, num_runs):
        total_time = 0
        total_memory = 0
        solutions_count = 0

        for i in range(num_runs):
            rows, cols, seed, cutting_rate, goal_and_start_spacing, lone_blocks_rate = self.generate_random_parameters()
            maze = Maze(rows=rows, cols=cols, seed=seed, lone_blocks_rate=lone_blocks_rate)
            environment = maze.generate_maze(rand=cutting_rate)
            start, goal = maze.set_start_and_goal(goal_and_start_spacing)
            ACTION_STEP = math.ceil(0.3*max(rows,cols))        

            # DO NOT UN-COMMENT IF YOUR NUM_RUN IS HIGH. Shows all plots visually for matrices generated
            """
            fig, ax = plt.subplots()
            ax.imshow(environment, cmap='Greys', origin='upper')
            ax.set_title(f"Run {i + 1}")
            plt.pause(1)  # Pause for a short duration to display the plot
            """
            density = self.calculate_obstacle_density(maze.matrix)
            path, exec_time, current, peak_memory, visited = self.run_search_algorithm(environment, start, goal, visualizing=False, action_step=ACTION_STEP)

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
        print(f"Average Peak Memory: {avg_memory/(1024 * 1024)} MB")
        print(f"Total Solutions: {solutions_count}")

    def euclidean_distance(self, rows, cols):
        distance = math.sqrt((rows-1 - 0)**2 + (cols-1 - 0)**2)
        return math.floor(distance)

    def get_next_actions(self, current, goal, environment, visited, action_step, radius):
        next_actions = []

        for dx, dy in self.directions_cost:
            last = None
            cost = self.directions_cost[(dx, dy)] 
            """
            TO IMPLEMENT. WILL WORK ON COST FOR DIRECTIONS LATER, FOR UCS
            """
            for i in range(1, action_step + 1):
                new_x, new_y = current[0] + i * dx, current[1] + i * dy

                if (new_x, new_y) in visited:  # If it was in visited, then skip
                    continue
                elif (new_x, new_y) == goal:
                    visited.add(last)
                    last = new_x, new_y
                    break
                elif self.is_valid(new_x, new_y, environment, radius) and (new_x, new_y) != goal:
                    visited.add(last)
                    last = (new_x, new_y)
                else:
                    break

            if last:
                next_actions.append(last)

        return next_actions

    def generate_random_parameters(self):
        rows = random.randint(15, 40)
        cols = random.randint(15, 40)
        seed = random.randint(1, 1000)
        cutting_rate = random.uniform(0.4, 0.85)
        goal_and_start_spacing = random.randint(round(0.25 * self.euclidean_distance(rows, cols)), round(0.85 * self.euclidean_distance(rows, cols)))
        lone_blocks_rate = random.uniform(0.9,1)
        return rows, cols, seed, cutting_rate, goal_and_start_spacing, lone_blocks_rate
    
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
    robot = Robot(ALGORITHM)
    random.seed()  # FOR REPRODUCIBILITY!

    # Single Run: Visualization
    if VISUALIZING and not STATIC:
        # Random Parameters
        rows, cols, seed, cutting_rate, goal_and_start_spacing, lone_blocks_rate = robot.generate_random_parameters()
        # ACTION_STEP = math.ceil(0.3*max(rows,cols))        

        robot.single_run(rows, cols, seed, cutting_rate, goal_and_start_spacing, ACTION_STEP)
    elif VISUALIZING and STATIC:
        # Static Parameters
        robot.single_run(ROWS, COLS, SEED, CUTTING_RATE, GOAL_AND_START_SPACING, LONE_BLOCKS_RATE, action_step=ACTION_STEP)
    else:
        # Multiple Runs: Average Scores
        robot.multiple_runs(NUM_RUNS)