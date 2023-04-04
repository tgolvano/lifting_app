# Based on the work by
# https://github.com/nicknochnack/MediaPipePoseEstimation
# Learning purposes



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
        'l_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER, 
        'l_hip': mp_pose.PoseLandmark.LEFT_HIP, 
        'l_knee': mp_pose.PoseLandmark.LEFT_KNEE,

    }

    lmk_coord = {}
    for lmk_type in landmark_types:
        lmk_coord[lmk_type] = extract_landmark_coordinates(landmark_types[lmk_type])
        
    angle_types = {
        'l_hip': calc_angle(lmk_coord['l_shoulder'], lmk_coord['l_hip'], lmk_coord['l_knee']),
    }

    return angle_types



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

            
    # plot values for each key in order of appearance


        except:
            pass

        # render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )



        cv2.imshow('MP Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

    # Plot values for each key in order of appearance
    for key in angles_over_time:
        plt.scatter(range(len(angles_over_time[key])), angles_over_time[key])
        plt.xlabel('Time frame')
        plt.ylabel(key)
        plt.title('Webster Snatch')
        plt.grid()
        plt.savefig(f"{key}_scatter.png")


