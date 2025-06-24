import cv2
import mediapipe as mp
import numpy as np
import math
import joblib
import pyttsx3
import csv
import os
from collections import Counter

# Function to calculate incenter
def incenter(a, b, c):
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    A = distance(b, c)
    B = distance(c, a)
    C = distance(a, b)
    P = A + B + C
    x = (A * a[0] + B * b[0] + C * c[0]) / P
    y = (A * a[1] + B * b[1] + C * c[1]) / P
    return (x, y)

# Euclidean distance function
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Load the model
model = joblib.load("best_posture_model_fixed_svm.pkl")

# Setup text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose()
face = mp_face.FaceMesh(refine_landmarks=True)

# Start webcam
cap = cv2.VideoCapture(0)

frame_no = 0
last_spoken = ""
recent_predictions = []
speak_interval = 150  # ~5 seconds assuming 30 FPS
csv_file = "posture_data.csv"

# Prepare CSV
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "F1 (Head Tilt)", "F2 (Slouch)", "Class"])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    h, w = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(image_rgb)
    face_results = face.process(image_rgb)

    predicted_class = ""
    F1 = F2 = None

    if pose_results.pose_landmarks and face_results.multi_face_landmarks:
        landmarks = {}
        pose_landmarks = pose_results.pose_landmarks.landmark
        landmarks['left_shoulder'] = (int(pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                                      int(pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        landmarks['right_shoulder'] = (int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                                       int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))

        face_landmarks = face_results.multi_face_landmarks[0].landmark

        def get_landmark(index):
            return (int(face_landmarks[index].x * w), int(face_landmarks[index].y * h))

        landmarks['left_ear'] = get_landmark(454)
        landmarks['right_ear'] = get_landmark(234)
        landmarks['left_eye'] = get_landmark(475)
        landmarks['right_eye'] = get_landmark(470)
        landmarks['nose'] = get_landmark(1)

        left_face_mid = incenter(landmarks['left_ear'], landmarks['left_eye'], landmarks['nose'])
        right_face_mid = incenter(landmarks['right_ear'], landmarks['right_eye'], landmarks['nose'])

        S1 = euclidean(left_face_mid, landmarks['left_shoulder'])
        S2 = euclidean(right_face_mid, landmarks['right_shoulder'])
        S3 = euclidean(landmarks['left_shoulder'], landmarks['right_shoulder'])

        if S3 != 0:
            F1 = (((S1 - S2) * 10) / S3)
            F2 = ((S1 + S2) / S3)

            # Predict class using model
            features = np.array([[F1, F2]])
            predicted_class = model.predict(features)[0]

            # Write to CSV
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_no, round(F1, 2), round(F2, 2)+0.12, predicted_class])

            # Collect predictions for 5-second interval
            if predicted_class:
                recent_predictions.append(predicted_class)

            # Speak once every 5 seconds
            if frame_no % speak_interval == 0 and recent_predictions:
                most_common = Counter(recent_predictions).most_common(1)[0][0]
                if most_common != "Looks good" and most_common != last_spoken:
                    engine.say(most_common)
                    engine.runAndWait()
                    last_spoken = most_common
                recent_predictions.clear()

            # Draw connections
            connections = [
                ('left_ear', 'left_eye'), ('left_ear', 'nose'), ('left_eye', 'nose'),
                ('nose', 'right_eye'), ('nose', 'right_ear'), ('right_eye', 'right_ear'),
                ('right_ear', 'right_shoulder'), ('right_shoulder', 'left_shoulder'), ('left_shoulder', 'left_ear')
            ]
            for a, b in connections:
                cv2.line(frame, landmarks[a], landmarks[b], (255, 0, 0), 2)

            # Draw landmarks
            for name, point in landmarks.items():
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
                cv2.putText(frame, name.replace("_", " "), (point[0] + 5, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Midpoints
            cv2.circle(frame, (int(left_face_mid[0]), int(left_face_mid[1])), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(right_face_mid[0]), int(right_face_mid[1])), 5, (0, 0, 255), -1)

            # Draw F1 and F2
            cv2.putText(frame, f"F1: {F1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"F2: {F2:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display predicted class
    if predicted_class:
        color = (0, 255, 0) if predicted_class == "Looks good" else (0, 0, 255)
        text_size = cv2.getTextSize(predicted_class, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = int((w - text_size[0]) / 2)
        cv2.putText(frame, predicted_class, (text_x, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Real-Time Posture Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
