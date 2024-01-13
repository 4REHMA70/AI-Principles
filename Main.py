import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from maze import Maze
from queue import PriorityQueue
import random
import math
import tracemalloc
import seaborn as sns
import time
# import ui
from config import *
import sys


class Robot:
    def __init__(self, algorithm='bfs', directions='8d', action_step=3, type='tree'):
        self.fig, self.ax = plt.subplots()
        
        self.algorithm = algorithm
        
        if directions == '8d':
            self.directions_cost = {(-1, 0): 1, (1, 0): 1, (0, -1): 1, (0, 1): 1, (-1, -1): 1, (-1, 1): 1, (1, -1): 1, (1, 1): 1}
        else: 
            self.directions_cost = {(-1, 0): 1, (1, 0): 1, (0, -1): 1, (0, 1): 1}

        self.action_step = action_step

        self.type = type

    def visualize(self, environment, paths=None, start=None, goal=None):

        self.ax.clear()  
        environment = np.array(environment, dtype=float)
                
        self.ax.imshow(environment, cmap='Greys', origin='upper')  
        if paths:
            
            for path in paths:
                path = np.array(path)
                self.ax.plot(path[:,1], path[:,0], marker='o', color='grey', linestyle='-.')

        if start:
            self.ax.plot(start[1], start[0], marker='h', color='blue', markersize=10, label='Start')

        if goal:
            self.ax.plot(goal[1], goal[0], marker='h', color='limegreen', markersize=10, label='Goal')

        self.ax.grid(True, linestyle='--', alpha=0.2)  # Uncomment this for grid lines
        
        self.ax.legend(loc='lower right', fontsize=6)   
        plt.pause(0.2)

        # ui_display = ui.UserInterface(np.array(environment))
        # ui_display.run()

    def depth_first_search(self, environment, start, goal, visualizing, radius=RADIUS, action_step=3):

        stack = [(start, [])]
        paths_explored = []
        visited = set()

        while stack:
            current, path = stack.pop() 
            distance = math.sqrt((goal[0] - current[0])**2 + (goal[1] - current[1])**2)
            if current == goal or distance < radius:
                paths_explored.append(path + [current])
                if visualizing:
                    self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                return path + [current], visited

            if self.type=='tree' and current in visited:
                continue

            visited.add(current)
            paths_explored.append(path + [current])

            if visualizing:
                self.visualize(environment, paths=paths_explored, start=start, goal=goal)

            for next_action, action_cost in self.get_next_actions(current, goal, path, environment, visited, action_step, radius):
                stack.append((next_action, path + [current]))

        return None
    
    def breadth_first_search(self, environment, start, goal, visualizing, radius=RADIUS, action_step=3):

        queue = deque([(start, [])])
        paths_explored = []
        visited = set()

        while queue:
            current, path = queue.popleft()
            
            distance = math.sqrt((goal[0] - current[0])**2 + (goal[1] - current[1])**2)

            if current == goal or distance < radius:
                paths_explored.append(path + [current])
                if visualizing:
                    self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                return path + [current], visited

            if self.type=='tree' and current in visited:
                continue

            visited.add(current)
            paths_explored.append(path + [current])

            if visualizing:
                self.visualize(environment, paths=paths_explored, start=start, goal=goal)

            """
            # Can test the action_step-code over here. Note: Step of the length of the maze is dangerous as it traverses in only that 
            action_step, consequently sacrificing exploration. 66% solutions missed 
            """

            for next_action, action_cost in self.get_next_actions(current=current, goal=goal, path=path, environment=environment, visited=visited, action_step=action_step, radius=radius):
                queue.append((next_action, path + [current])) # Appending only last values in each direction to queue 
                
        return None

    def uniform_cost_search(self, environment, start, goal, visualizing, radius=RADIUS, action_step=3):
        
        open_set = PriorityQueue()
        open_set.put((0, start, []))
        visited = set()
        paths_explored = []

        while not open_set.empty():
            cost, current, path = open_set.get()

            if current == goal:
                paths_explored.append(path + [current])  
                if visualizing:
                    self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                return path + [current], visited

            if self.type=='tree' and current in visited:
                continue
                
            visited.add(current)
            paths_explored.append(path + [current])
            
            if visualizing:
                self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                # current, goal, environment, visited, action_step, radius
            for next_node, action_cost in self.get_next_actions(current, goal, path, environment, visited, action_step, radius):
                new_cost = cost + action_cost  
                open_set.put((new_cost, next_node, path + [current]))
        
        return None
    
    def a_star_search(self, environment, start, goal, visualizing, radius=RADIUS, action_step=3):

        open_set = [(start, 0, math.sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2), [])]
        visited = set()
        paths_explored = []

        while open_set:
            open_set.sort(key=lambda x: x[1] + x[2])  # Sorting by cost + heuristic
            current, cost, _, path = open_set.pop(0)

            if current == goal:
                paths_explored.append(path + [current])
                if visualizing:
                    self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                return path + [current], visited

            if self.type=='tree' and current in visited:
                continue

            visited.add(current)
            paths_explored.append(path + [current])

            if visualizing:
                self.visualize(environment, paths=paths_explored, start=start, goal=goal)

            for next_action, action_cost in self.get_next_actions(current=current, goal=goal, path=path, environment=environment, visited=visited, action_step=action_step, radius=radius):
                new_cost = cost + action_cost

                # Euclidean Distance
                # heuristic_value = math.sqrt((goal[0] - current[0]) ** 2 + (goal[1] - current[1]) ** 2)

                # Manhattan Distance
                heuristic_value = abs(goal[0] - current[0]) + abs(goal[1] - current[1])  

                open_set.append((next_action, new_cost, heuristic_value, path + [current]))

        return None

    def iterative_deepening_search(self, environment, start, goal, visualizing, radius=RADIUS, max_depth=1000, action_step=3):
        for depth in range(max_depth+1):
            result = self.depth_limited_search(environment, start, goal, visualizing, radius, depth, action_step)
            if result:
                return result

        return None

    def depth_limited_search(self, environment, start, goal, visualizing, radius, depth_limit, action_step=3):
        # Exact same logic but follows until distance reaches depth limit. Difference is this function is called iteratively till max depth in IDS
        stack = [(start, [])]
        paths_explored = []
        visited = set()

        while stack:
            current, path = stack.pop()

            distance = math.hypot((goal[0] - current[0]), (goal[1] - current[1]))
            # distance = math.sqrt((goal[0] - current[0])**2 + (goal[1] - current[1])**2)

            if current == goal or distance < radius:
                paths_explored.append(path + [current])
                if visualizing:
                    self.visualize(environment, paths=paths_explored, start=start, goal=goal)
                return path + [current], visited

            if self.type=='tree' and current in visited or len(path)>= depth_limit:
                continue

            visited.add(current)
            paths_explored.append(path + [current])

            if visualizing:
                self.visualize(environment, paths=paths_explored, start=start, goal=goal)
            # Action cost isn't used, so excluded
            for next_action, action_cost in self.get_next_actions(current, goal, path, environment, visited, action_step, radius):
                stack.append((next_action, path + [current]))

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
        elif self.algorithm=='ucs':
            result = self.uniform_cost_search(environment, start, goal, visualizing, action_step=action_step)
        elif self.algorithm=='ids':
            result = self.iterative_deepening_search(environment, start, goal, visualizing, action_step=action_step)
        else:
            print('Invalid algorithm name. Valid names are: bfs, dfs, a_star, ucs')
            result = None

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
        # Test to prove that the radius checking for goal works. Or to prove that it can't cut corners!
        environment=np.zeros((10,10), dtype=int)
        # environment[:1, :] = 1
        # environment[-1:, :] = 1
        # environment[:, :1] = 1
        # environment[:, -1:] = 1
        # environment[1,6], environment[6,1] = 1, 1 # TESTING FOR RADIUS OF 3. start, goal = (7,7), (1,1)
        environment[0,3], environment[3,0] = 1, 1 # TESTING FOR RADIUS OF 2. start, goal = (7,7), (0,0)
        action_step = 3
        start, goal = (7,7), (0,0) 
        """
        # CAN MANUALLY SKIP VISUALIZING HERE IF WANT ONLY METRICS AND LAST PATH
        path, execution_time, current, peak_memory, visited = self.run_search_algorithm(environment, start, goal, visualizing=VISUALIZING, action_step=action_step)
        
        if path is not None:

            for _ in path:
                self.visualize(environment, paths=[path], start=start, goal=goal)
            plt.pause(0.1)
            plt.close()
            print(self.algorithm, self.type)
            print("Path found!")
            print(path)

        else:
            plt.pause(0.1)
            print("No solution found!")

    def multiple_runs(self, num_runs):
        total_time = 0
        total_memory = 0
        solutions_count = 0
        all_exec_times = []
        all_peak_memories = []
        all_path_to_spacing_ratios = []
        all_search_space_coverages = []

        # outfile = open('output.log', 'w')
        # sys.stdout = outfile

        for i in range(num_runs):
            rows, cols, seed, cutting_rate, goal_and_start_spacing, lone_blocks_rate = self.get_random_parameters()
            # rows,cols=50,25
            maze = Maze(rows=rows, cols=cols, seed=seed, lone_blocks_rate=lone_blocks_rate)
            environment = maze.generate_maze(rand=cutting_rate)
            start, goal = maze.set_start_and_goal(goal_and_start_spacing)
            # Can comment out if wanting a static action step
            action_step = math.ceil(0.3*max(rows,cols))        

            # DO NOT UN-COMMENT IF YOUR NUM_RUN IS HIGH. Shows all plots visually for matrices generated (not the searches)
            """
            fig, ax = plt.subplots()
            ax.imshow(environment, cmap='Greys', origin='upper')
            ax.set_title(f"Run {i + 1}")
            plt.pause(1)  # Pause for a short duration to display the plot
            """
            density = self.calculate_obstacle_density(maze.matrix)
            path, exec_time, current, peak_memory, visited = self.run_search_algorithm(environment, start, goal, visualizing=False, action_step=action_step)

            # Because path and visited may be None when radius too big to explore
            if path is not None and visited is not None:
                path_to_spacing_ratio = len(path)/goal_and_start_spacing # The smaller, the better (generally)
                search_space_coverage = len(visited)/(rows*cols)
                solutions_count += 1            
                all_exec_times.append(exec_time)
                all_peak_memories.append(peak_memory)
                all_path_to_spacing_ratios.append(path_to_spacing_ratio)
                all_search_space_coverages.append(search_space_coverage)

            else:
                path_to_spacing_ratio = None # For when there's no path due to radius size
                search_space_coverage = 0

            total_time += exec_time
            total_memory += peak_memory
            print('\n',i)

            array = [rows, cols, seed, cutting_rate, path, lone_blocks_rate, goal_and_start_spacing, density, path_to_spacing_ratio, search_space_coverage*100]
            for label, value in zip(["rows", "cols", "seed", "cutting_rate", "path", "lone_blocks_rate", "goal_and_start_spacing", "density", "path_to_spacing_ratio", "search_space_coverage (%)"], array):
                print(f"{label}: {value}")

        avg_time = total_time/num_runs
        avg_memory = total_memory/num_runs
        variance_exec_time = np.var(all_exec_times)
        variance_peak_memory = np.var(all_peak_memories)

        print()
        print(self.algorithm, self.type)
        print(f"Total Number of runs: {num_runs}")
        print(f"Average Execution Time: {avg_time} seconds")
        print(f"Average Peak Memory: {avg_memory / (1024 * 1024)} MB")
        print(f"Variance of Execution Time: {variance_exec_time} seconds^2")
        print(f"Variance of Peak Memory: {variance_peak_memory / (1024 * 1024)} MB^2")
        print(f"Average Path to Spacing Ratio: {np.mean(all_path_to_spacing_ratios)}")
        print(f"Average Search Space Coverage: {np.mean(all_search_space_coverages)}")
        print(f"Total Solutions: {solutions_count}")

        # Distribution plot for execution time
        plt.figure(figsize=(10, 5))
        sns.histplot(all_exec_times, kde=True, color='skyblue', bins=20)
        plt.title(f'Distribution of Execution Time for {robot.algorithm}')
        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Frequency')
        plt.axvline(x=avg_time, color='red', linestyle='--', linewidth=1, label=f'Avg: {avg_time:.2f}')

        plt.show()

        # Distribution plot for peak memory
        plt.figure(figsize=(10, 5))
        sns.histplot(all_peak_memories, kde=True, color='salmon', bins=20)
        plt.title(f'Distribution of Peak Memory for {robot.algorithm}')
        plt.xlabel('Peak Memory (MB)')
        plt.ylabel('Frequency')
        plt.axvline(x=avg_memory, color='red', linestyle='--', linewidth=1, label=f'Avg: {avg_memory:.2f}')

        plt.show()
        
        # outfile.close()

    def euclidean_distance(self, rows, cols):
        distance = math.sqrt((rows-1 - 0)**2 + (cols-1 - 0)**2)
        return math.floor(distance)

    def get_next_actions(self, current, goal, path, environment, visited, action_step, radius):
        next_actions = []
        action_costs = []
        diagonals = {
            (-1, -1): ((1, 0), (0, 1)), 
            (-1, 1): ((1, 0), (0, -1)), 
            (1, -1): ((-1, 0), (0, 1)), 
            (1, 1): ((-1, 0), (0, -1)) 
        }        
        skip_direction = False

        for direction in self.directions_cost:
            last = None
            cost = self.directions_cost[direction]

            for i in range(1, action_step + 1):
                new_node = current[0] + i * direction[0], current[1] + i * direction[1]
                
                # CUTTING CORNERS LOGIC
                # THIS CHECKING MORE THAN TRIPLES THE DURATION FOR THE ALGORITHMS (3.5x). Should be after visited check
                if direction in diagonals:
                    # Get the adjacent blocks for the current diagonal direction
                    diagonal_opposite1, diagonal_opposite2 = diagonals[direction]
                    # Calculate the coordinates of the adjacent blocks
                    new_node_adjacent_block1 = (new_node[0] + diagonal_opposite1[0], new_node[1] + diagonal_opposite1[1])
                    new_node_adjacent_block2 = (new_node[0] + diagonal_opposite2[0], new_node[1] + diagonal_opposite2[1])
                    
                    # Check if any of the adjacent blocks is invalid
                    if not (self.is_valid(new_node_adjacent_block1,environment=environment,radius=radius)) or not (self.is_valid(new_node_adjacent_block2,environment=environment,radius=radius)): 
                        skip_direction = True
                        last = None
                        continue

                # THESE TWO CHECKS NEED TO BE BELOW CUTTING CORNERS LOGIC. SOMEHOW INTERRUPTS IT OTHERWISE

                if new_node in visited and self.type=='tree':
                    continue
 
                if (len(path)>2) and new_node == path[-1]: # TO PREVENT IT FROM GOING BACK TO PARENT IMMEDIATELY
                    continue

                elif new_node == goal:
                    visited.add(last)
                    last = new_node
                    break
                elif self.is_valid(new_node, environment, radius) and new_node != goal:
                    visited.add(last)
                    last = new_node
                else:
                    break

            if skip_direction:
                continue

            if last:
                next_actions.append(last)
                action_costs.append(cost)

        return zip(next_actions, action_costs)

    def is_valid(self, new_node, environment, radius):
        radius -= 1
        for i in range(max(0, new_node[0]-radius), min(len(environment), new_node[0]+radius+1)):
            for j in range(max(0, new_node[1]-radius), min(len(environment[0]), new_node[1]+radius+1)):
                if environment[i][j] == 1:
                    return False 
        return (
            0 <= new_node[0] < len(environment) 
            and 0 <= new_node[1] < len(environment[0])
            and environment[new_node[0]][new_node[1]] == 0
        )

    def get_random_parameters(self):
        rows = random.randint(15, 30)
        cols = random.randint(15, 30)
        # HIGH ROWS AND COLS HERE LIKE 40 QUADRATICALLY INCR. SEARCH SPACE. STOPS OUTPUT
        seed = random.randint(1, 1000)
        cutting_rate = random.uniform(0.4, 0.85)
        # goal_and_start_spacing = random.randint(round(0.25 * self.euclidean_distance(rows, cols)), round(0.85 * self.euclidean_distance(rows, cols)))
        # SOMETIMES GOAL START SPACING IS PROBLEMATIC
        goal_and_start_spacing = 10
        lone_blocks_rate = random.uniform(0.9,1)
        return rows, cols, seed, cutting_rate, goal_and_start_spacing, lone_blocks_rate

if __name__ == "__main__":
    
    # for ALGORITHM in ['bfs', 'a_star', 'ucs', 'ids', 'dfs']:
    #     for TYPE in ['tree','graph']:
    robot = Robot(ALGORITHM, DIRECTIONS, ACTION_STEP, type=TYPE)
    random.seed()  # FOR REPRODUCIBILITY!

    if (RADIUS < 1) or (GOAL_AND_START_SPACING > min(ROWS, COLS)):
        print("Error: Change values. RADIUS > 1, and GOAL_AND_START_SPACING > the smaller of ROWS and COLS. ROW AND COLS > 5")
    else:
        # Single Run: Visualization
        if SINGLE and not STATIC:
            # Random Parameters
            rows, cols, seed, cutting_rate, goal_and_start_spacing, lone_blocks_rate = robot.get_random_parameters()
            # ACTION_STEP = math.ceil(0.3*max(rows,cols))
            robot.single_run(rows, cols, seed, cutting_rate, goal_and_start_spacing, action_step=robot.action_step)

        elif SINGLE and STATIC:
            # Static Parameters
            robot.single_run(ROWS, COLS, SEED, CUTTING_RATE, GOAL_AND_START_SPACING, LONE_BLOCKS_RATE, action_step=robot.action_step)
        else:
            # Multiple Runs: Average Scores
            robot.multiple_runs(NUM_RUNS)