import torch

import cv2
import mediapipe as mp
import numpy as np
from numpy import linalg as LA

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




cap = cv2.VideoCapture(0)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
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

            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)


            # Visualize angle
            cv2.putText(image, str(l_hip_angle), 
                           tuple(np.multiply(l_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(r_hip_angle), 
                           tuple(np.multiply(r_hip, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
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
            
    cap.Release()
    cv2.destroyAllWindows()
    


