import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# Load model and label encoder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dict = joblib.load(os.path.join(BASE_DIR, "models", "trained_model.pkl"))
model = model_dict["model"]
le = model_dict["label_encoder"]

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Helper to extract pose landmarks
def extract_pose_landmarks(results):
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()

# Open webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting real-time prediction...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image and convert color
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Predict
    pose_vector = extract_pose_landmarks(results)
    if pose_vector is not None:
        pose_vector = pose_vector.reshape(1, -1)
        pred = model.predict(pose_vector)[0]
        label = le.inverse_transform([pred])[0]
        cv2.putText(frame, f"Shot: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show frame
    cv2.imshow("Cricket Shot Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
