# src/map_clusters_to_shots.py
import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import cdist
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_FEATURES = os.path.join(BASE, "data", "pose_features.csv")             # labeled images (label column)
CLUSTERED = os.path.join(BASE, "data", "pose_sequences_clustered.csv")    # clustered per-frame CSV (must have cluster)
OUT_JSON = os.path.join(BASE, "data", "cluster_to_shot_mapping.csv")

def method_centroid_matching(df_labeled, df_clustered):
    """
    Method A: centroid matching
    - compute mean vector per label (from labeled dataset)
    - compute centroid per cluster (from clustered frames)
    - compute Euclidean distance and map clusters to nearest label
    """
    X_lab = df_labeled.drop(columns=["label"], errors="ignore").values
    labels = df_labeled["label"].values
    unique_labels = sorted(df_labeled["label"].unique())

    mean_per_label = {}
    for lbl in unique_labels:
        mean_per_label[lbl] = df_labeled[df_labeled["label"] == lbl].drop(columns=["label"]).mean().values

    label_means_mat = np.stack([mean_per_label[l] for l in unique_labels])

    cluster_map = {}
    for cluster_id in sorted(df_clustered["cluster"].unique()):
        cluster_vectors = df_clustered[df_clustered["cluster"] == cluster_id].drop(columns=["video_name", "frame_id", "cluster"], errors="ignore").values
        centroid = cluster_vectors.mean(axis=0)
        dists = cdist([centroid], label_means_mat, metric="euclidean")[0]
        best_label = unique_labels[int(np.argmin(dists))]
        cluster_map[cluster_id] = {
            "method": "centroid",
            "assigned_label": best_label,
            "distances": {unique_labels[i]: float(dists[i]) for i in range(len(unique_labels))}
        }
    return cluster_map

def method_nn_majority(df_labeled, df_clustered, k=1):
    """
    Method B: nearest neighbor majority voting
    - For each clustered frame, find nearest labeled sample(s) in the labeled pool (k-NN).
    - Assign each clustered frame the label of its nearest labeled neighbor.
    - For each cluster, take the mode of assigned labels.
    Note: This can be slow if dataset is large. Consider sampling or indexing (KDTree).
    """
    from sklearn.neighbors import NearestNeighbors

    X_lab = df_labeled.drop(columns=["label"], errors="ignore").values
    y_lab = df_labeled["label"].values

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_lab)

    cluster_map = {}
    df_features_clustered = df_clustered.drop(columns=["video_name", "frame_id", "cluster"], errors="ignore")
    X_clu = df_features_clustered.values

    # Query in batches to avoid memory issues
    distances, indices = nbrs.kneighbors(X_clu, return_distance=True)
    # map each clustered frame to label (if k>1, choose nearest)
    nearest_labels = y_lab[indices[:, 0]]
    df_clustered_copy = df_clustered.copy()
    df_clustered_copy["nn_label"] = nearest_labels

    for cluster_id in sorted(df_clustered_copy["cluster"].unique()):
        sub = df_clustered_copy[df_clustered_copy["cluster"] == cluster_id]
        mode_label, count = Counter(sub["nn_label"]).most_common(1)[0]
        cluster_map[cluster_id] = {
            "method": "nn_majority",
            "assigned_label": mode_label,
            "counts": dict(Counter(sub["nn_label"])),
            "support": int(len(sub))
        }
    return cluster_map

def main():
    print("[INFO] Loading datasets...")
    df_labeled = pd.read_csv(POSE_FEATURES)
    df_clustered = pd.read_csv(CLUSTERED)

    # Sanity: ensure feature dims match (drop meta cols)
    feat_cols_labeled = [c for c in df_labeled.columns if c != "label"]
    feat_cols_clustered = [c for c in df_clustered.columns if c not in ("video_name", "frame_id", "cluster")]

    if len(feat_cols_labeled) != len(feat_cols_clustered):
        print("[ERROR] Feature dimensionality mismatch!")
        print("Labeled features:", len(feat_cols_labeled))
        print("Clustered features:", len(feat_cols_clustered))
        # attempt to align by position: trim or pad if safe (but ideally ensure same extraction method)
        return

    print("[INFO] Running centroid matching...")
    centroid_map = method_centroid_matching(df_labeled, df_clustered)

    print("[INFO] Running nearest-neighbor majority voting (this may take some time)...")
    nn_map = method_nn_majority(df_labeled, df_clustered, k=1)

    # Combine and save mapping
    rows = []
    for cluster_id in sorted(df_clustered["cluster"].unique()):
        r = {
            "cluster_id": int(cluster_id),
            "centroid_assigned_label": centroid_map[cluster_id]["assigned_label"],
            "centroid_distances": centroid_map[cluster_id]["distances"],
            "nn_assigned_label": nn_map[cluster_id]["assigned_label"],
            "nn_support": nn_map[cluster_id]["support"],
            "nn_counts": nn_map[cluster_id]["counts"]
        }
        rows.append(r)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_JSON, index=False)
    print(f"[SUCCESS] Mapping saved at: {OUT_JSON}")
    print(out_df.to_string(index=False))

if __name__ == "__main__":
    main()
