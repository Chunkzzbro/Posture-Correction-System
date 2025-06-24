# 🧍‍♂️ Real-Time Posture Correction System

A computer vision-based system that monitors user posture in real-time using webcam input, classifies postures using extracted landmarks and SVM, and provides auditory feedback to promote ergonomic health.

---

## 📖 Abstract

This project implements a real-time posture correction system using computer vision and a lightweight SVM model. It processes live video feeds, detects body and facial landmarks via MediaPipe, extracts key features, and classifies user posture into three categories. The system provides immediate auditory feedback, helping users self-correct poor posture habits.

---

## 💡 Motivation

Poor sitting posture is increasingly common in today's digital environment, leading to musculoskeletal issues. This project aims to build a dynamic, user-friendly tool to monitor posture continuously and promote healthier habits through real-time alerts.

---

## ❗ Problem Statement

Most existing applications only provide static posture assessments. This system addresses the need for:
- Continuous, real-time posture analysis
- Lightweight feedback mechanisms
- Efficient and responsive performance

---

## ⚙️ Methodology

### 1. Data Collection
- Used the [posture_correction_v4](https://universe.roboflow.com/posturecorrection/posture_correction_v4) dataset (~4700 images)
- Labeled into:
  - “Looks Good”
  - “Sit Up Straight”
  - “Straighten Head”

### 2. MediaPipe Pipeline
- Used MediaPipe Pose and Face Mesh
- Detected:
  - 33 pose landmarks
  - 468 facial landmarks
- Real-time processing at high FPS

### 3. Feature Extraction
- Calculated custom metrics:
  - S1, S2: distances from facial midpoints to shoulders
  - S3: shoulder-to-shoulder distance
- Derived features:
  - `F1 = ((S1 - S2) × 10) / S3` → head tilt
  - `F2 = (S1 + S2) / S3` → head proximity
- Visualized landmarks and feature lines in real-time

### 4. Model Development
- Used Support Vector Machine (SVM) classifier
- Input: F1 and F2 features
- Trained on labeled image data

### 5. Real-Time Processing
- OpenCV-based live feed
- MediaPipe for landmark detection
- Integrated classifier for posture prediction

### 6. Feedback Mechanism
- Text-to-speech audio alerts:
  - “Sit up straight”
  - “Straighten your head”
- Logs posture over time for user review

## 📊 Results

| Posture Class        | F1-F2 Range                 | Accuracy         |
|----------------------|-----------------------------|------------------|
| Looks Good           | F1 ≈ 0, F2 > 1.48            | High             |
| Sit Up Straight      | F1 ≠ 0, F2 < 1.48            | High             |
| Straighten Head      | F1 >> 0 or << 0, F2 normal   | Moderate–High    |

- **Training Accuracy:** 68%  
- **Test Accuracy:** 53%  
- **Live Performance:** Responsive, ~30 FPS

---

## 🎥 Demo

📺 **YouTube Demo Video**: [Click here to watch](https://youtu.be/SRiI0eHTCrQ)

---

## 📝 Documentation

📄 **Project Notes and Design on Notion**: [View Notion Page](https://erratic-herring-227.notion.site/1d734ef4af9c80c98166def1dbf2c64e?v=1d734ef4af9c81a98135000ce724b654)

---

