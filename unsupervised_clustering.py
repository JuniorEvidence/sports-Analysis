import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Paths
DATA_PATH = "../data/pose_sequences.csv"
MODEL_PATH = "../models/clustering_model.pkl"
os.makedirs("../models", exist_ok=True)

print("[INFO] Loading pose data...")
data = pd.read_csv(DATA_PATH)

# Drop metadata columns
X = data.drop(columns=["video_name", "frame_id"], errors='ignore')

print(f"[INFO] Loaded data shape: {X.shape}")

# 1️⃣ Normalize the data
scaler = StandardScaler()

# 2️⃣ Dimensionality reduction
pca = PCA(n_components=3, random_state=42)

# 3️⃣ Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

# Combine into pipeline
pipeline = Pipeline([
    ("scaler", scaler),
    ("pca", pca),
    ("kmeans", kmeans)
])

print("[INFO] Fitting clustering pipeline...")
pipeline.fit(X)

# Save model
joblib.dump(pipeline, MODEL_PATH)
print(f"[SUCCESS] Model saved at: {MODEL_PATH}")

# Transform data for visualization
X_pca = pipeline.named_steps["pca"].transform(scaler.fit_transform(X))
labels = pipeline.named_steps["kmeans"].labels_

# Create visualization DataFrame
vis_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
vis_df["Cluster"] = labels

# Visualization
sns.set(style="whitegrid", rc={"figure.figsize": (10, 7)})
sns.scatterplot(data=vis_df, x="PC1", y="PC2", hue="Cluster", palette="tab10")
plt.title("Unsupervised Clustering of Batting Pose Patterns")
plt.savefig("../data/pose_clusters.png", dpi=300)
plt.show()

# Add clusters to the original data
data["cluster"] = labels
data.to_csv("../data/pose_sequences_clustered.csv", index=False)
print("[INFO] Clustered data saved to: ../data/pose_sequences_clustered.csv")
