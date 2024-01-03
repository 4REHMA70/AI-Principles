from Main import Robot
import numpy as np
from maze import Maze
import matplotlib.pyplot as plt
import seaborn as sns

import math
import random

def sensitivity_analysis(robot, parameter_to_vary, values_to_try, fixed_params, num_runs):
    all_exec_times = []
    all_peak_memory = []

    for value in values_to_try:
        current_params = fixed_params.copy()
        current_params[parameter_to_vary] = value

        for _ in range(num_runs):
            rows, cols, seed, cutting_rate, goal_and_start_spacing, lone_blocks_rate = current_params.values()

            maze = Maze(rows=rows, cols=cols, seed=seed, lone_blocks_rate=lone_blocks_rate)
            environment = maze.generate_maze(rand=cutting_rate)
            start, goal = maze.set_start_and_goal(goal_and_start_spacing)
            ACTION_STEP = math.ceil(0.3 * max(rows, cols))

            path, exec_time, current, peak_memory, visited = robot.run_search_algorithm(
                environment, start, goal, visualizing=False, action_step=ACTION_STEP
            )

            if path is not None and visited is not None:
                all_exec_times.append(exec_time)
                all_peak_memory.append(peak_memory)

    plot_distribution_plots(all_exec_times, all_peak_memory, parameter_to_vary, robot.algorithm)


def plot_distribution_plots(all_exec_times, all_peak_memory, param, algorithm):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    sns.histplot(all_exec_times, kde=True, color='blue', ax=axes[0])
    axes[0].set_title(f"{algorithm} - Execution Time Distribution Plot ({param})")
    axes[0].set_xlabel("Execution Time (s)")
    axes[0].set_ylabel("Density")

    sns.histplot(all_peak_memory, kde=True, color='orange', ax=axes[1])
    axes[1].set_title(f"{algorithm} - Peak Memory Distribution Plot ({param})")
    axes[1].set_xlabel("Peak Memory (MB)")
    axes[1].set_ylabel("Density")

    fig.tight_layout()
    plt.show()

robot = Robot('dfs')
random.seed()  # FOR REPRODUCIBILITY!

fixed_params = {
    "rows": 20,
    "cols": 20,
    "seed": 123,
    "cutting_rate": 0.7,
    "goal_and_start_spacing": 10,
    "lone_blocks_rate": 0.95
}

values_to_try = [0.1, 0.3, 0.5, 0.7, 0.9]  # Adjust based on the parameter you want to vary
num_runs = 5  # number of runs 

sensitivity_analysis(robot, "cutting_rate", values_to_try, fixed_params, num_runs)
