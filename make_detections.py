# Based on the work by
# https://github.com/nicknochnack/MediaPipePoseEstimation
# Learning purposes


import time
import json
import cv2
import mediapipe as mp
import numpy as np
from numpy import linalg as LA
import os
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calc_angle(a, b, c):
    
    # a, b and c in 3D cartesian coordinates
    a = np.array(a)
    b = np.array(b) # vertex
    c = np.array(c)

    # Obtains angle from dot product
    vector_ba = np.subtract(a, b)
    vector_bc = np.subtract(c, b)
    norm_ba = LA.norm(vector_ba)
    norm_bc = LA.norm(vector_bc)
    dot_product = np.dot(vector_ba, vector_bc)

    angle_radians = np.arccos(dot_product / (norm_ba * norm_bc))

    # rounded to nearest integer 360 degree
    angle = np.round(np.abs(angle_radians * 180.0 / np.pi)) 

    return angle


def extract_angles_from_landmarks(results):
    landmarks = results.pose_landmarks.landmark

    def extract_landmark_coordinates(lmk):
        return [landmarks[lmk.value].x, landmarks[lmk.value].y]
    
    landmark_types = {
        'l_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
        'l_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
        'l_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER, 
        'l_hip': mp_pose.PoseLandmark.LEFT_HIP, 
        'l_knee': mp_pose.PoseLandmark.LEFT_KNEE,
        'l_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
        'l_foot': mp_pose.PoseLandmark.LEFT_FOOT_INDEX,

        'r_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
        'r_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
        'r_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER, 
        'r_hip': mp_pose.PoseLandmark.RIGHT_HIP, 
        'r_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
        'r_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
        'r_foot': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    }

    lmk_coord = {}
    for lmk_type in landmark_types:
        lmk_coord[lmk_type] = extract_landmark_coordinates(landmark_types[lmk_type])
        
    angle_types = {
        'l_elbow': calc_angle(lmk_coord['l_shoulder'], lmk_coord['l_elbow'], lmk_coord['l_wrist']),
        'r_elbow': calc_angle(lmk_coord['r_shoulder'], lmk_coord['r_elbow'], lmk_coord['r_wrist']),
        'l_shoulder': calc_angle(lmk_coord['l_hip'], lmk_coord['l_shoulder'], lmk_coord['l_elbow']),
        'r_shoulder': calc_angle(lmk_coord['r_hip'], lmk_coord['r_shoulder'], lmk_coord['r_elbow']),
        'l_hip': calc_angle(lmk_coord['l_shoulder'], lmk_coord['l_hip'], lmk_coord['l_knee']),
        'r_hip': calc_angle(lmk_coord['r_shoulder'], lmk_coord['r_hip'], lmk_coord['r_knee']),
        'l_knee': calc_angle(lmk_coord['l_hip'], lmk_coord['l_knee'], lmk_coord['l_ankle']),
        'r_knee': calc_angle(lmk_coord['r_hip'], lmk_coord['r_knee'], lmk_coord['r_ankle']),
        'l_ankle': calc_angle(lmk_coord['l_knee'], lmk_coord['l_ankle'], lmk_coord['l_foot']),
        'r_ankle': calc_angle(lmk_coord['r_knee'], lmk_coord['r_ankle'], lmk_coord['r_foot']),
    }

    return angle_types

def scatter_joint_angles(angles_over_time: dict, title: str) -> None:
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
        ax.set_title(f"{key[2:].capitalize()} {title}")
        ax.legend()
        ax.grid()
        plt.savefig(f"{key[2:]}_{title}_scatter.png")




start_t = time.time()

# VIDEO MP4 format

# Set the path to your mp4 video
with open('config.json') as f:
    config = json.load(f)
video_path = os.path.join(config['data_path'], "003.mp4")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")




# Initialization of a dictionary to contain the values of the angles over the video
angles_over_time = {}

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, smooth_landmarks=True) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # If video ends or fails, GTFO
        if not ret:
            break
        if frame is None:
            continue


        # channels order change
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # detection
        results = pose.process(image)

        # channels to BGR sorting
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # landmarks and angles extraction
        try:
            angle_types = extract_angles_from_landmarks(results)
            
            for key, value in angle_types.items():
                if key not in angles_over_time:
                    angles_over_time[key] = np.array([value])
                else:
                    angles_over_time[key] = np.append(angles_over_time[key], value)
        except:
            print("Some error happened retrieving angles from landmarks")
            pass

        # render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        

        #cv2.imshow('MP Feed', image)

       # if cv2.waitKey(10) & 0xFF == ord('q'):
       #     break
            
    cap.release()
    cv2.destroyAllWindows()

    # Plot values for each key in order of appearance
    scatter_joint_angles(angles_over_time, "Webster_snatch")

end_t = time.time()
elapsed_t = end_t - start_t

print(f"Elapsed time: {elapsed_t}")