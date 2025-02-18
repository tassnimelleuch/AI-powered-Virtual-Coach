import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model # type: ignore

# Load the model
model = load_model("../models/yoga_pose_model.h5")

# Define pose labels (update this list based on your dataset)
pose_labels = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to classify pose
def classify_pose(keypoints, model):
    keypoints = np.array(keypoints).flatten().reshape(1, -1)
    prediction = model.predict(keypoints)
    return np.argmax(prediction)  # Returns the ID of the predicted pose

# Main function
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from webcam.")
            break

        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Debug: Check the shape of the image
        print("Image shape:", image_rgb.shape)

        # Process the frame
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Extract keypoints
            keypoints = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]

            # Classify the pose
            pose_id = classify_pose(keypoints, model)  # Get the predicted pose ID
            pose_name = pose_labels[pose_id]  # Map the ID to the pose name

            # Display the pose name on the screen
            cv2.putText(frame, f"Pose: {pose_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Yoga Pose Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()