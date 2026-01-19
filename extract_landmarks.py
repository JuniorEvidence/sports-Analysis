import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset\dataset")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "data", "pose_features1.csv")

# Label mapping from folder name to readable label
label_map = {
    "drive": "drive",
    "legglance-flick": "leg_glance_flick",
    "pullshot": "pull_shot",
    "sweep": "sweep",
    "no_shot":"no_shot"
}

# Store pose data
landmarks_list = []
labels = []

def extract_pose_landmarks(image):
    """Extracts 33 pose landmarks for a single image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        return keypoints
    else:
        return None

# Read all images and extract landmarks
print("[INFO] Starting landmark extraction...")
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    label = label_map.get(folder)
    if not label:
        print(f"[WARNING] Unknown folder skipped: {folder}")
        continue

    print(f"[INFO] Processing {folder}...")
    for img_file in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_file)
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
            keypoints = extract_pose_landmarks(image)
            if keypoints:
                landmarks_list.append(keypoints)
                labels.append(label)
        except Exception as e:
            print(f"[ERROR] Failed to process {img_file}: {e}")
def normalize_landmarks(landmarks):
    # Extract shoulders
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_dist = np.linalg.norm(np.array([left_shoulder.x, left_shoulder.y]) -
                                   np.array([right_shoulder.x, right_shoulder.y]))
    
    # Normalize all landmarks by shoulder width
    normalized = []
    for lm in landmarks:
        normalized.extend([
            (lm.x - left_shoulder.x) / shoulder_dist,
            (lm.y - left_shoulder.y) / shoulder_dist,
            (lm.z) / shoulder_dist,
            lm.visibility
        ])
    return normalized

def is_standing_pose(landmarks, threshold=0.03):
    """
    Determine if the given pose is a standing pose based on simple heuristics.
    - Calculates vertical distance between shoulders and wrists.
    - Checks minimal horizontal spread.
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # Heuristic 1: Both wrists are close (in X-axis) to shoulders → arms down
    wrist_close = (abs(left_shoulder.x - left_wrist.x) < threshold and
                   abs(right_shoulder.x - right_wrist.x) < threshold)

    # Heuristic 2: Wrists are below shoulders → arms are not raised
    wrist_below = (left_wrist.y > left_shoulder.y and right_wrist.y > right_shoulder.y)

    return wrist_close and wrist_below
# Convert to DataFrame and Save
print("[INFO] Saving landmark data to CSV...")
df = pd.DataFrame(landmarks_list)
df['label'] = labels
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"[SUCCESS] Saved extracted features to: {OUTPUT_CSV_PATH}")
