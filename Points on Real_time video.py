import cv2
import mediapipe as mp
import numpy as np
import math

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

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose()
face = mp_face.FaceMesh(refine_landmarks=True)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(image_rgb)
    face_results = face.process(image_rgb)

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

        # Calculate midpoints
        left_face_mid = incenter(landmarks['left_ear'], landmarks['left_eye'], landmarks['nose'])
        right_face_mid = incenter(landmarks['right_ear'], landmarks['right_eye'], landmarks['nose'])

        # Distance-based metrics
        S1 = euclidean(left_face_mid, landmarks['left_shoulder'])
        S2 = euclidean(right_face_mid, landmarks['right_shoulder'])
        S3 = euclidean(landmarks['left_shoulder'], landmarks['right_shoulder'])

        if S3 != 0:
            F1 = ((S1 - S2) * 10) / S3
            F2 = (S1 + S2) / S3

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

            # Draw midpoints
            cv2.circle(frame, (int(left_face_mid[0]), int(left_face_mid[1])), 5, (0, 0, 255), -1)

            cv2.circle(frame, (int(right_face_mid[0]), int(right_face_mid[1])), 5, (0, 0, 255), -1)

            # Draw F1 and F2
            cv2.putText(frame, f"F1: {F1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"F2: {F2:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize output
    resized_frame = cv2.resize(frame, (int(w), int(h)))
    cv2.imshow("Real-Time Posture Tracking", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
