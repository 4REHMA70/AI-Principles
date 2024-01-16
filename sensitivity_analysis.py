from Main import Robot
import numpy as np
from maze import Maze
import matplotlib.pyplot as plt
import seaborn as sns
import math

def sensitivity_analysis(robot, varied_params, fixed_params, num_runs):
    with open('sensitivity_output.log', 'w') as output:
        output.write("Algorithm,Parameter,Value,Avg Execution Time (s),Std Dev Execution Time (s),"
        "Avg Peak Memory (MB),Std Dev Peak Memory (MB),Avg Coeff of Variation Execution Time,"
        "Avg Coeff of Variation Peak Memory\n")

        for algorithm in ['bfs', 'dfs', 'a_star', 'ucs', 'ids']:
            robot.algorithm = algorithm
            print('___________________________________________')
            print(algorithm.upper())
            print('___________________________________________\n')
            for param_name, values_to_try in varied_params.items():
                all_exec_times = []
                all_peak_memories = []
                print()
                print('Current param: ', param_name)
                print('Values: ', values_to_try)
                print()
                for value in values_to_try:
                    current_params = fixed_params.copy()
                    current_params[param_name] = value
                    print(current_params)
                    for _ in range(num_runs):
                        maze = Maze(rows=current_params["rows"], cols=current_params["cols"], seed=current_params["seed"], lone_blocks_rate=current_params["lone_blocks_rate"])
                        environment = maze.generate_maze(cutting_rate=current_params["cutting_rate"])
                        start, goal = maze.set_start_and_goal(current_params["goal_and_start_spacing"])
                        action_step = current_params["action_step"]

                        path, exec_time, current, peak_memory, visited = robot.run_search_algorithm(environment, start, goal, visualizing=False, action_step=action_step)

                        if path is not None and visited is not None:
                            all_exec_times.append(exec_time)
                            all_peak_memories.append(peak_memory/(1024 * 1024))

                avg_exec_time = np.mean(all_exec_times)
                std_dev_exec_time = np.std(all_exec_times)
                avg_peak_memory = np.mean(all_peak_memories)
                std_dev_peak_memory = np.std(all_peak_memories)

                coeff_of_variation_exec_time = std_dev_exec_time/avg_exec_time
                coeff_of_variation_peak_memory = std_dev_peak_memory/avg_peak_memory

                output.write(f"{algorithm},{param_name},{values_to_try},{round(avg_exec_time, 4)},{round(std_dev_exec_time, 4)},"f"{round(avg_peak_memory, 4)},{round(std_dev_peak_memory, 4)},{round(coeff_of_variation_exec_time, 4)},"f"{round(coeff_of_variation_peak_memory, 4)}\n")

                
                # COMMENT THIS OUT IF YOU WANT ONLY OUTPUTS
                plot_distribution_plots(all_exec_times, param_name, avg_exec_time, algorithm, f'Rows={fixed_params["rows"]}, Cols={current_params["cols"]}', 'Avg Execution Time (s)')
                plot_distribution_plots(all_peak_memories, param_name, avg_peak_memory, algorithm, f'Rows={fixed_params["rows"]}, Cols={current_params["cols"]}', 'Avg Peak Memory (MB)')


def plot_distribution_plots(data, param, avg_value, algorithm, rows_cols_str, label):
    plt.figure(figsize=(8, 5))
    sns.histplot(data, kde=True, color='skyblue', bins=20)
    plt.title(f'{algorithm} - Distribution of {label}\n({param})')
    plt.xlabel(label)
    plt.ylabel('Frequency')

    plt.axvline(x=avg_value, color='red', linestyle='--', linewidth=1, label=f'Avg: {avg_value:.2f}')
    plt.show()


fixed_params = {
    "rows": 50,
    "cols": 50,
    "seed": 123,
    "cutting_rate": 0.6,
    "goal_and_start_spacing": 10,
    "lone_blocks_rate": 0.95,
    "action_step": 3
}

varied_params = {
    "cutting_rate": [0.1, 0.3, 0.5, 0.7, 0.9],
    "lone_blocks_rate": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    "directions": ['8d', '4d'],
    "action_step": [1, 2, 3, 10],
    "radius": [1, 2]
}

num_runs = 100
robot = Robot()
print("OUTPUT DISPLAYED IN SENSITIVITY_OUTPUT.LOG")
sensitivity_analysis(robot, varied_params, fixed_params, num_runs)
