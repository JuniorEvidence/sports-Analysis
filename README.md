# sports-Analysis
STEP 1: Extract Pose Landmarks from Images
python src/extract_landmarks.py


📌 Output:

data/pose_features.csv

🔹 STEP 2: Train Supervised Shot Classifier
python src/train_model.py


📌 Output:

models/trained_model.pkl

Classification report in terminal

🔹 STEP 3: Real-Time Shot Prediction
python src/predict_live.py


📌 Uses webcam + MediaPipe to classify shots live.
