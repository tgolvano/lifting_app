import numpy as np
import matplotlib.pyplot as plt

angles_over_time = {}

for i in range(10):
    angle_types = {
        'l_hip': 5*i,
        'r_hip': 6*i,
    }
    for key, value in angle_types.items():
        if key not in angles_over_time:
            angles_over_time[key] = np.array([value])
        else:
            angles_over_time[key] = np.append(angles_over_time[key], value)

# Checks the resulting dictionary
print(angles_over_time)

for key in angles_over_time:
    plt.scatter(range(len(angles_over_time[key])), angles_over_time[key])
    plt.xlabel('Time frame')
    plt.ylabel(key)
    plt.grid()
    plt.savefig(f"{key}_scatter.png")