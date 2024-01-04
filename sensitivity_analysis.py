from Main import Robot
import numpy as np
from maze import Maze
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random

def sensitivity_analysis(robot, varied_params, fixed_params, num_runs):
    with open('sensitivity_output.log', 'w') as output:
        output.write("Algorithm,Parameter,Value,Avg Execution Time (s),Std Dev Execution Time (s),"
        "Avg Peak Memory (MB),Std Dev Peak Memory (MB),Avg Coeff of Variation Execution Time,"
        "Avg Coeff of Variation Peak Memory\n")

        for algorithm in ['ucs', 'bfs', 'dfs', 'a_star']:
            robot.algorithm = algorithm
            for param_name, values_to_try in varied_params.items():
                all_exec_times = []
                all_peak_memories = []

                for value in values_to_try:
                    current_params = fixed_params.copy()
                    current_params[param_name] = value

                    for _ in range(num_runs):
                        maze = Maze(rows=current_params["rows"], cols=current_params["cols"],
                                    seed=current_params["seed"], lone_blocks_rate=current_params["lone_blocks_rate"])
                        environment = maze.generate_maze(rand=current_params["cutting_rate"])
                        start, goal = maze.set_start_and_goal(current_params["goal_and_start_spacing"])
                        ACTION_STEP = math.ceil(0.3 * max(current_params["rows"], current_params["cols"]))

                        path, exec_time, current, peak_memory, visited = robot.run_search_algorithm(environment, start, goal, visualizing=False, action_step=ACTION_STEP)

                        if path is not None and visited is not None:
                            all_exec_times.append(exec_time)
                            all_peak_memories.append(peak_memory)

                avg_exec_time = np.mean(all_exec_times)
                std_dev_exec_time = np.std(all_exec_times)
                avg_peak_memory = np.mean(all_peak_memories)
                std_dev_peak_memory = np.std(all_peak_memories)

                coeff_of_variation_exec_time = std_dev_exec_time / avg_exec_time
                coeff_of_variation_peak_memory = std_dev_peak_memory / avg_peak_memory

                output.write(f"{algorithm},{param_name},{values_to_try},{round(avg_exec_time, 6)},{round(std_dev_exec_time, 6)},"
                f"{round(avg_peak_memory, 6)},{round(std_dev_peak_memory, 6)},{round(coeff_of_variation_exec_time, 6)},"
                f"{round(coeff_of_variation_peak_memory, 6)}\n")

                plot_distribution_plots(all_exec_times, param_name, avg_exec_time, algorithm, f'Rows={fixed_params["rows"]}, Cols={current_params["cols"]}', 'Avg Execution Time (s)')
                plot_distribution_plots(all_peak_memories, param_name, avg_peak_memory, algorithm, f'Rows={fixed_params["rows"]}, Cols={current_params["cols"]}', 'Avg Peak Memory (MB)')


def plot_distribution_plots(data, param, avg_value, algorithm, rows_cols_str, y_label):
    plt.figure(figsize=(8, 5))
    sns.histplot(data, kde=True, color='skyblue', bins=20)
    plt.title(f'{algorithm} - Distribution of {y_label}\n({param}, {rows_cols_str})')
    plt.xlabel(param)
    plt.ylabel('Frequency')

    plt.axvline(x=avg_value, color='red', linestyle='--', linewidth=1, label=f'Avg: {avg_value:.2f}')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


fixed_params = {
    "rows": 50,
    "cols": 50,
    "seed": 123,
    "cutting_rate": 0.7,
    "goal_and_start_spacing": 10,
    "lone_blocks_rate": 0.95
}

varied_params = {
    "cutting_rate": [0.1, 0.3, 0.5, 0.7, 0.9],
    # "rows": [50, 25],
    "lone_blocks_rate": [0.1, 0.5, 0.95],
    "directions": ['8d', '4d'],
    "action_step": [1, 2, 3],
    "radius": [1, 2]
}

num_runs = 100
robot = Robot()
sensitivity_analysis(robot, varied_params, fixed_params, num_runs)


"""
Cutting Rate:

UCS and BFS: Increasing cutting_rate generally leads to a decrease in average execution time but an increase in standard deviation.
DFS and A:* The influence is less pronounced in A*, but increasing cutting_rate still results in a decrease in average execution time with an increase in standard deviation.
Lone Blocks Rate:

UCS: The effect is not very significant.
BFS and DFS: Lower lone_blocks_rate tends to result in lower average execution time with a moderate increase in standard deviation.
A:* Lower lone_blocks_rate tends to result in lower average execution time with a moderate increase in standard deviation.
Directions:

UCS, BFS, DFS, and A:* '8d' generally performs better than '4d' in terms of both average execution time and standard deviation.
Action Step:

UCS, BFS, DFS, and A:* Smaller action_step tends to result in better performance for all algorithms.
Radius:

UCS, BFS, DFS, and A:* The radius parameter does not seem to strongly influence the performance of any of the algorithms.



For ucs and bfs, the cutting_rate and lone_blocks_rate parameters have the largest impact on execution time and memory usage variability. The coefficients of variation for those parameters are much higher than other parameters.
For dfs, the cutting_rate and lone_blocks_rate parameters also have a sizable impact on variability. However, the memory usage shows extremely high variation even for parameters like directions and radius.
For A*, the radius, action_step and directions show very little variation in memory usage. But the execution time variation is extremely high across all parameters - over 2x the coefficient of variation in many cases.

"""