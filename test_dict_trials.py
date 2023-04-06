import numpy as np
import matplotlib.pyplot as plt

angles_over_time = {}

for i in range(10):
    angle_types = {
        'l_hip': np.ones((10,)) * 5*i,
        'r_hip': np.ones((13,)) * 6*i,
        'l_elbow': np.ones((7,)) * 6*i,
        'r_elbow': np.ones((5,)) * 3/(i+0.1),
    }

    for key, value in angle_types.items():
        if key not in angles_over_time:
            angles_over_time[key] = np.array([value])
        else:
            angles_over_time[key] = np.append(angles_over_time[key], value)


# Create a dictionary of joint groups
group_joints = {}
for key in angles_over_time.keys():
    joint = key[2:] # Get the name of the joint by removing the first 2 characters
    if joint not in group_joints:
        group_joints[joint] = [key]
    else:
        group_joints[joint].append(key)

# Plot the figures for each group
for group in group_joints.values():
    fig, ax = plt.subplots()
    for key in group:
        if key[0] == 'l':
            ax.scatter(range(len(angles_over_time[key])), angles_over_time[key], label=key, color='red')
        else: # Starts with 'r'
            ax.scatter(range(len(angles_over_time[key])), angles_over_time[key], label=key, color='blue')

    ax.set_xlabel('Time frame')
    ax.set_ylabel('Joint angle')
    ax.set_title(f"{key[2:].capitalize()}")
    ax.legend()
    ax.grid()
    plt.show()