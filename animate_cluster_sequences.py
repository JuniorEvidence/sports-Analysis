# src/animate_cluster_sequences.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse

# MediaPipe landmark connections (used to draw skeleton): list of pairs
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (15,17),(16,18),(17,19),(18,20)
    # This is a reduced set; MediaPipe provides a full set, but these illustrate skeleton links.
]

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTERED = os.path.join(BASE, "data", "pose_sequences_clustered.csv")
OUT_DIR = os.path.join(BASE, "data", "animations")
os.makedirs(OUT_DIR, exist_ok=True)

def find_segments(df, cluster_id):
    """
    Find contiguous segments where df.cluster == cluster_id and frame_id increments by 1.
    Returns list of (video_name, start_idx_in_df, end_idx_in_df, length)
    """
    dfc = df[df["cluster"] == cluster_id].copy().reset_index()
    segments = []
    if dfc.empty:
        return segments
    # Group by video_name
    for vid, group in dfc.groupby("video_name"):
        # sort by frame_id
        group_sorted = group.sort_values("frame_id").reset_index()
        start_idx = group_sorted.loc[0, "index"]
        prev_frame = group_sorted.loc[0, "frame_id"]
        seg_start = 0
        for i in range(1, len(group_sorted)):
            cur_frame = group_sorted.loc[i, "frame_id"]
            if cur_frame == prev_frame + 1:
                prev_frame = cur_frame
                continue
            else:
                seg_end = i - 1
                length = seg_end - seg_start + 1
                segments.append((vid, group_sorted.loc[seg_start, "index"], group_sorted.loc[seg_end, "index"], length))
                seg_start = i
                prev_frame = cur_frame
        # final segment
        seg_end = len(group_sorted) - 1
        length = seg_end - seg_start + 1
        segments.append((vid, group_sorted.loc[seg_start, "index"], group_sorted.loc[seg_end, "index"], length))
    # sort by length desc
    segments_sorted = sorted(segments, key=lambda x: x[3], reverse=True)
    return segments_sorted

def get_landmarks_from_row(row):
    # row contains flattened landmarks followed by cluster, video_name, frame_id (we drop meta)
    # first columns are x0,y0,z0,vis0, x1,y1,z1,vis1, ...
    # Return numpy array of shape (num_landmarks, 4)
    # Find feature columns by dropping known meta columns
    meta_cols = {"video_name", "frame_id", "cluster"}
    feat_cols = [c for c in row.index if c not in meta_cols]
    vals = row[feat_cols].values.astype(float)
    num_landmarks = len(vals) // 4
    return vals.reshape((num_landmarks, 4))

def animate_segment(df, start_idx, end_idx, out_file, fps=15, figsize=(6,8)):
    segment = df.iloc[start_idx:end_idx+1].reset_index(drop=True)
    frames = []
    for i in range(len(segment)):
        lm = get_landmarks_from_row(segment.loc[i])
        frames.append(lm)  # shape (num_landmarks, 4)

    num_landmarks = frames[0].shape[0]
    # Use normalized coordinates on plot: x in [0,1], y in [0,1] (Mediapipe uses normalized coords)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # invert y for image-like coordinates
    ax.set_xticks([])
    ax.set_yticks([])
    scat = ax.scatter([], [], s=30)
    lines = []
    for _ in POSE_CONNECTIONS:
        line, = ax.plot([], [], lw=2)
        lines.append(line)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        for ln in lines:
            ln.set_data([], [])
        return [scat] + lines

    def update(frame_idx):
        lm = frames[frame_idx]
        xy = lm[:, :2]
        scat.set_offsets(xy)
        # update connections
        for i, (a, b) in enumerate(POSE_CONNECTIONS):
            if a < num_landmarks and b < num_landmarks:
                x_vals = [xy[a, 0], xy[b, 0]]
                y_vals = [xy[a, 1], xy[b, 1]]
                lines[i].set_data(x_vals, y_vals)
            else:
                lines[i].set_data([], [])
        return [scat] + lines

    anim = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True)
    writer = FFMpegWriter(fps=fps)
    anim.save(out_file, writer=writer)
    plt.close(fig)
    print(f"[SAVED] {out_file}")

def main(cluster_id=0, n_segments=3):
    print("[INFO] Loading clustered pose data...")
    df = pd.read_csv(CLUSTERED)
    df_columns = df.columns.tolist()
    if "cluster" not in df_columns:
        print("[ERROR] 'cluster' column not found in clustered data.")
        return

    segments = find_segments(df, cluster_id)
    if not segments:
        print(f"[WARN] No segments found for cluster {cluster_id}")
        return

    # Select top n_segments
    chosen = segments[:n_segments]
    for idx, (video_name, start_idx, end_idx, length) in enumerate(chosen):
        out_file = os.path.join(OUT_DIR, f"cluster{cluster_id}_seg{idx}_{video_name}_len{length}.mp4")
        animate_segment(df, start_idx, end_idx, out_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=int, default=0, help="cluster id to animate")
    parser.add_argument("--n", type=int, default=3, help="how many top segments to animate")
    args = parser.parse_args()
    main(cluster_id=args.cluster, n_segments=args.n)
