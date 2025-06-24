import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import joblib

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=True)
face = mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True)

# Incenter calculation
def incenter(a, b, c):
    def distance(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))
    A, B, C = distance(b, c), distance(c, a), distance(a, b)
    P = A + B + C
    x = (A * a[0] + B * b[0] + C * c[0]) / P
    y = (A * a[1] + B * b[1] + C * c[1]) / P
    return (x, y)

# Feature extraction
def calculate_features(pose_landmarks, face_landmarks, image_shape):
    h, w = image_shape
    try:
        left_shoulder = (int(pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        right_shoulder = (int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        left_ear = (int(face_landmarks[454].x * w), int(face_landmarks[454].y * h))
        right_ear = (int(face_landmarks[234].x * w), int(face_landmarks[234].y * h))
        left_eye = (int(face_landmarks[475].x * w), int(face_landmarks[475].y * h))
        right_eye = (int(face_landmarks[470].x * w), int(face_landmarks[470].y * h))
        nose = (int(face_landmarks[1].x * w), int(face_landmarks[1].y * h))

        left_face_mid = incenter(left_ear, left_eye, nose)
        right_face_mid = incenter(right_ear, right_eye, nose)

        def euclidean(p1, p2): return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        S1 = euclidean(left_face_mid, left_shoulder)
        S2 = euclidean(right_face_mid, right_shoulder)
        S3 = euclidean(left_shoulder, right_shoulder)

        F1 = ((S1 - S2) * 10) / S3
        F2 = (S1 + S2) / S3

        return [F1, F2]
    except:
        return None

# Load dataset
def load_data(folder):
    X, y = [], []
    labels = os.listdir(folder)
    for label in tqdm(labels, desc=f"Loading {os.path.basename(folder)}"):
        class_path = os.path.join(folder, label)
        if not os.path.isdir(class_path): continue
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            image = cv2.imread(img_path)
            if image is None: continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image_rgb)
            face_results = face.process(image_rgb)
            if pose_results.pose_landmarks and face_results.multi_face_landmarks:
                features = calculate_features(pose_results.pose_landmarks.landmark,
                                              face_results.multi_face_landmarks[0].landmark,
                                              image.shape[:2])
                if features:
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)

# Paths
base_path = "posture_correction_v4.v2i.folder"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "valid")
test_path = os.path.join(base_path, "test")

# Load datasets
X_train, y_train = load_data(train_path)
X_val, y_val = load_data(val_path)
X_total = np.vstack((X_train, X_val))
y_total = np.hstack((y_train, y_val))

# Cross-validation with fixed params
k_values = range(2, 16)
accuracies = []
best_acc = 0
best_model = None
best_k = None

print("\n--- Cross-Validation with fixed SVM (RBF, C=10, gamma=1) ---\n")

for k in k_values:
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, val_idx in kf.split(X_total, y_total):
        X_tr, X_va = X_total[train_idx], X_total[val_idx]
        y_tr, y_va = y_total[train_idx], y_total[val_idx]

        model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma=1))
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        acc = accuracy_score(y_va, preds)
        fold_accuracies.append(acc)

    avg_acc = np.mean(fold_accuracies)
    accuracies.append(avg_acc)

    if avg_acc > best_acc:
        best_acc = avg_acc
        best_model = model
        best_k = k

    print(f"k = {k} | Avg Accuracy: {avg_acc:.4f}")

print(f"\n✅ Best k = {best_k} with Avg Accuracy = {best_acc:.4f}")

# Save best model
model_filename = 'best_posture_model_fixed_svm.pkl'
joblib.dump(best_model, model_filename)
print(f"✅ Best model saved to {model_filename}")

# Evaluate on test data
test_predictions, test_labels, test_images = [], [], []
for label in tqdm(os.listdir(test_path), desc="Testing"):
    label_path = os.path.join(test_path, label)
    if not os.path.isdir(label_path): continue
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        image = cv2.imread(img_path)
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        face_results = face.process(image_rgb)
        if pose_results.pose_landmarks and face_results.multi_face_landmarks:
            features = calculate_features(pose_results.pose_landmarks.landmark,
                                          face_results.multi_face_landmarks[0].landmark,
                                          image.shape[:2])
            if features:
                pred = best_model.predict([features])[0]
                test_predictions.append(pred)
                test_labels.append(label)
                test_images.append((image, label, pred))

print("\n📊 Test Classification Report:")
print(classification_report(test_labels, test_predictions))
print("✅ Test Accuracy:", accuracy_score(test_labels, test_predictions))

# Show sample predictions
sample_indices = random.sample(range(len(test_images)), min(5, len(test_images)))
fig, axs = plt.subplots(1, len(sample_indices), figsize=(18, 5))
for i, idx in enumerate(sample_indices):
    img, true_label, pred_label = test_images[idx]
    axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
    axs[i].axis('off')
plt.tight_layout()
plt.show()

# Plot accuracy vs k
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.xlabel("k (Number of Folds)")
plt.ylabel("Average Cross-Validation Accuracy")
plt.title("Accuracy vs k-fold")
plt.grid(True)
plt.xticks(k_values)
plt.show()
