import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define paths
VIDEO_DIR = "../dataset/crickshot"
OUTPUT_CSV = "../data/pose_sequences.csv"

def extract_pose_from_video(video_path, video_name):
    """
    Extracts pose landmarks from every frame in a video.
    Returns: list of flattened landmark vectors.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    pose_vectors = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_vector = []
            for lm in landmarks:
                frame_vector.extend([lm.x, lm.y, lm.z, lm.visibility])
            pose_vectors.append(frame_vector)
        frame_count += 1

    cap.release()

    if pose_vectors:
        df = pd.DataFrame(pose_vectors)
        df["video_name"] = video_name
        df["frame_id"] = np.arange(len(df))
        return df
    else:
        return None


def main():
    all_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".avi", ".mov"))]
    all_data = []

    print(f"[INFO] Found {len(all_videos)} videos to process...")

    for video in tqdm(all_videos, desc="Extracting Pose Sequences"):
        video_path = os.path.join(VIDEO_DIR, video)
        video_name = os.path.splitext(video)[0]

        df = extract_pose_from_video(video_path, video_name)
        if df is not None:
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"[SUCCESS] Pose sequences saved to: {OUTPUT_CSV}")
    else:
        print("[WARNING] No pose data extracted from any video.")


if __name__ == "__main__":
    main()
