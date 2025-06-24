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


# Load image
image_path = 'posture_correction_v4.v2i.folder/train/sit up straight/extract0001_jpg.rf.0a6b511f137696c1944c2b047fc72d4b.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=True)
face = mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True)


# Get pose and face results
pose_results = pose.process(image_rgb)
face_results = face.process(image_rgb)

# Get required landmarks
h, w = image.shape[:2]
landmarks = {}

# Pose landmarks (shoulders)
pose_landmarks = pose_results.pose_landmarks.landmark
landmarks['left_shoulder'] = (int(pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                              int(pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
landmarks['right_shoulder'] = (int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                               int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))

# Face landmarks (ears, eyes, nose)
face_landmarks = face_results.multi_face_landmarks[0].landmark


def get_landmark(index):
    return (int(face_landmarks[index].x * w), int(face_landmarks[index].y * h))


landmarks['left_ear'] = get_landmark(454)
landmarks['right_ear'] = get_landmark(234)
landmarks['left_eye'] = get_landmark(475)
landmarks['right_eye'] = get_landmark(470)
landmarks['nose'] = get_landmark(1)

# Calculate midpoints (incenter)
left_face_mid = incenter(landmarks['left_ear'], landmarks['left_eye'], landmarks['nose'])
right_face_mid = incenter(landmarks['right_ear'], landmarks['right_eye'], landmarks['nose'])


# Distance calculations
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


S1 = euclidean(left_face_mid, landmarks['left_shoulder'])
S2 = euclidean(right_face_mid, landmarks['right_shoulder'])
S3 = euclidean(landmarks['left_shoulder'], landmarks['right_shoulder'])

F1 = ((S1 - S2) * 10) / S3
F2 = (S1 + S2) / S3

# Draw connections
connections = [
    ('left_ear', 'left_eye'), ('left_ear', 'nose'), ('left_eye', 'nose'),
    ('nose', 'right_eye'), ('nose', 'right_ear'), ('right_eye', 'right_ear'),
    ('right_ear', 'right_shoulder'), ('right_shoulder', 'left_shoulder'), ('left_shoulder', 'left_ear')
]

for a, b in connections:
    pt1 = landmarks[a]
    pt2 = landmarks[b]
    cv2.line(image, pt1, pt2, (255, 0, 0), 2)

# Draw points and text
# Draw points and text for each landmark
for name, point in landmarks.items():
    cv2.circle(image, point, 5, (0, 255, 0), -1)
    cv2.putText(image, name.replace("_", " "), (point[0] - 30, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

# Draw face midpoints
cv2.circle(image, (int(left_face_mid[0]), int(left_face_mid[1])), 5, (0, 0, 255), -1)

cv2.circle(image, (int(right_face_mid[0]), int(right_face_mid[1])), 5, (0, 0, 255), -1)

# Show F1 and F2
cv2.putText(image, f"F1: {F1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(image, f"F2: {F2:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Resize image to half the original size
resized_image = cv2.resize(image, (int(w * 0.7), int(h * 0.7)))

# Display resized image
cv2.imshow("Posture Landmarks (Resized)", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
