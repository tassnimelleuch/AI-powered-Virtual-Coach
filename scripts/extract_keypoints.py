import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Function to extract keypoints
def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        return np.array(keypoints).flatten()  # Flatten to 1D array
    else:
        return None

# Main function to process the dataset
def process_dataset(data_dir):
    data = []
    labels = []

    for pose_name in os.listdir(data_dir):
        pose_dir = os.path.join(data_dir, pose_name)
        if not os.path.isdir(pose_dir):
            continue

        for image_name in os.listdir(pose_dir):
            image_path = os.path.join(pose_dir, image_name)
            keypoints = extract_keypoints(image_path)

            if keypoints is not None:
                data.append(keypoints)
                labels.append(pose_name)

    return np.array(data), np.array(labels)

# Save the dataset
def save_dataset(data, labels, output_file):
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv(output_file, index=False)

# Run the script
if __name__ == "__main__":
    data_dir = "../data"  # Path to the dataset
    output_file = "../data/yoga_poses_keypoints.csv"

    data, labels = process_dataset(data_dir)
    save_dataset(data, labels, output_file)
    print(f"Dataset saved to {output_file}")