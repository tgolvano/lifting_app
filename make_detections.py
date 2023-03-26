# Based on the work by
# https://github.com/nicknochnack/MediaPipePoseEstimation
# Learning purposes



import torch
import json
import cv2
import mediapipe as mp
import numpy as np
from numpy import linalg as LA
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    
    # a, b and c in 3D cartesian coordinates
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # More accurate for angles closer to 90 degrees
    radians = np.arccos(np.dot(np.subtract(a, b), np.subtract(c, b)) / (LA.norm(np.subtract(a, b)) * LA.norm(np.subtract(c, b))))

    # More accurate for angles closer to 0 or 180
    #radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    # rounded to nearest integer
    angle = np.round(np.abs(radians * 180.0/np.pi)) 

    return angle

options = ['WebCam', 'mp4 video']

user_input = ''
input_message = "Pick an option:\n"

for index, item in enumerate(options):
    input_message += f'{index+1}) {item}\n'

while user_input not in map(str, range(1, len(options) + 1)):
    user_input = input(input_message)

print('You picked: ' + options[int(user_input) - 1])

# CAM
if int(user_input) == 1:
    cap = cv2.VideoCapture(0)

# VIDEO MP4 format
elif int(user_input) == 2:
    # Set the path to your mp4 video
    with open('config.json') as f:
        config = json.load(f)
    video_path = os.path.join(config['data_path'], "001.mp4")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
else:
    print("How did you even got here?\n")



#

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, smooth_landmarks=True) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # channels order change
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # detection
        results = pose.process(image)

        # channels to BGR sorting
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # landmarks extraction
        try:
            landmarks = results.pose_landmarks.landmark
          
            # Coordinates of landmarks
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            l_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            r_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]


# HIP
            l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
# KNEE
            l_knee_angle = calculate_angle(l_ankle, l_knee, l_hip)
            r_knee_angle = calculate_angle(r_ankle, r_knee, r_hip)
# ELBOW
            l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
# SHOULDER
            l_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
            r_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
# ANKLE DORSIFLEXION
            l_ankle = calculate_angle(l_knee, l_ankle, l_foot)
            r_ankle = calculate_angle(r_knee, r_ankle, r_foot)



            # Visualize angle
            cv2.putText(image, str(l_hip_angle), 
                           tuple(np.multiply(l_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(r_hip_angle), 
                           tuple(np.multiply(r_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

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
    


