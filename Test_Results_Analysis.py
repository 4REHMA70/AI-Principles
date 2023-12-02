import numpy as np
from scipy.stats import sem, t

# Replace these with your actual data
execution_times = [0.001598, 0.003852, 0.002052, 0.006160, 0.000765, 0.005421]
peak_memory = [0.028881, 0.100863, 0.036677, 0.174265, 0.019989, 0.134560]

# Calculate mean
mean_execution_time = np.mean(execution_times)
mean_peak_memory = np.mean(peak_memory)

# Calculate standard deviation
std_execution_time = np.std(execution_times)
std_peak_memory = np.std(peak_memory)

# Calculate coefficient of variation
cv_execution_time = (std_execution_time / mean_execution_time) * 100
cv_peak_memory = (std_peak_memory / mean_peak_memory) * 100

# Calculate confidence intervals (95% confidence)
confidence_interval_execution_time = t.interval(0.95, len(execution_times)-1, loc=mean_execution_time, scale=sem(execution_times))
confidence_interval_peak_memory = t.interval(0.95, len(peak_memory)-1, loc=mean_peak_memory, scale=sem(peak_memory))

print("Execution Time:")
print(f"Mean: {mean_execution_time}")
print(f"Standard Deviation: {std_execution_time}")
print(f"Coefficient of Variation: {cv_execution_time}%")
print(f"Confidence Interval: {confidence_interval_execution_time}")

print("\nPeak Memory:")
print(f"Mean: {mean_peak_memory}")
print(f"Standard Deviation: {std_peak_memory}")
print(f"Coefficient of Variation: {cv_peak_memory}%")
print(f"Confidence Interval: {confidence_interval_peak_memory}")

"""
Execution Time:
Mean: 0.003308
Standard Deviation: 0.001994287424954922
Coefficient of Variation: 60.286802447246735%
Confidence Interval: (0.001015368880032725, 0.005600631119967276)

Peak Memory:
Mean: 0.08253916666666666
Standard Deviation: 0.05823890309296889
Coefficient of Variation: 70.55911204939339%
Confidence Interval: (0.015587773425686402, 0.14949055990764692)
"""