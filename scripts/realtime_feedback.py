import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = load_model("../models/yoga_pose_model.h5")

# Define pose labels (update this list based on your dataset)
pose_labels = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cosine_angle = dot_product / (magnitude_ab * magnitude_bc)
    angle = math.degrees(math.acos(cosine_angle))
    return angle

# Function to classify pose
def classify_pose(keypoints, model):
    keypoints = np.array(keypoints).flatten().reshape(1, -1)
    prediction = model.predict(keypoints)
    return np.argmax(prediction)  

# Function to provide feedback based on pose landmarks
def provide_feedback(landmarks, pose_name):
    feedback = []
        
    if pose_name == "Downdog":
        # Extract landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # Calculate angles
        shoulder_hip_knee_angle_left = calculate_angle(left_shoulder, left_hip, left_knee)
        shoulder_hip_knee_angle_right = calculate_angle(right_shoulder, right_hip, right_knee)
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
        back_angle = calculate_angle(left_wrist, left_shoulder, left_hip)
        heel_distance = abs(left_ankle.y - right_ankle.y)
        shoulder_hip_angle = calculate_angle(left_wrist, left_shoulder, left_hip)

        # Define thresholds for a perfect pose
        hip_angle_lower_threshold = 160
        hip_angle_upper_threshold = 175
        arm_angle_threshold = 170
        leg_angle_threshold = 170
        back_angle_threshold = 170
        heel_distance_threshold = 0.05
        shoulder_hip_angle_threshold = 105

        # Check if the pose is perfect
        is_perfect = (
            hip_angle_lower_threshold <= shoulder_hip_knee_angle_left <= hip_angle_upper_threshold and
            hip_angle_lower_threshold <= shoulder_hip_knee_angle_right <= hip_angle_upper_threshold and
            left_arm_angle >= arm_angle_threshold and
            right_arm_angle >= arm_angle_threshold and
            left_leg_angle >= leg_angle_threshold and
            right_leg_angle >= leg_angle_threshold and
            back_angle >= back_angle_threshold and
            heel_distance <= heel_distance_threshold and
            shoulder_hip_angle <= shoulder_hip_angle_threshold
        )

        if is_perfect:
            feedback.append("Perfect pose!")
        else:
            # Provide feedback only if the pose is not perfect
            if shoulder_hip_knee_angle_left < hip_angle_lower_threshold or shoulder_hip_knee_angle_right < hip_angle_lower_threshold:
                feedback.append("Lower your hips slightly to create an inverted 'V' shape.")
            elif shoulder_hip_knee_angle_left > hip_angle_upper_threshold or shoulder_hip_knee_angle_right > hip_angle_upper_threshold:
                feedback.append("Lift your hips higher to maintain the proper downward dog shape.")

            if left_arm_angle < arm_angle_threshold or right_arm_angle < arm_angle_threshold:
                feedback.append("Straighten your arms to maintain alignment.")

            if left_leg_angle < leg_angle_threshold or right_leg_angle < leg_angle_threshold:
                feedback.append("Straighten your legs and engage your thighs.")

            if back_angle < back_angle_threshold:
                feedback.append("Extend your spine and push your chest towards your thighs.")

            if heel_distance > heel_distance_threshold:
                feedback.append("Balance your weight evenly between both feet.")
            

    elif pose_name == "Plank":
        # Feedback for Plank pose
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # 1. Check full-body alignment (shoulder-hip-ankle straightness)
        left_body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        right_body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)

        if left_body_angle < 170 or right_body_angle < 170:
            feedback.append("Keep your body straight from shoulders to ankles. Avoid sagging your hips.")
        elif left_body_angle > 180 or right_body_angle > 180:
            feedback.append("Lower your hips slightly to maintain a straight line.")

        # 2. Check if hips are too high or too low
        hip_height = (left_hip.y + right_hip.y) / 2
        shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
        ankle_height = (left_ankle.y + right_ankle.y) / 2

        if hip_height < (shoulder_height + ankle_height) / 2 - 0.05:
            feedback.append("Lower your hips slightly to align with your shoulders and ankles.")
        elif hip_height > (shoulder_height + ankle_height) / 2 + 0.05:
            feedback.append("Raise your hips slightly to maintain a neutral spine.")

        # 3. Check core engagement (prevent arching back)
        spine_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        if spine_angle < 175:
            feedback.append("Engage your core to keep your lower back neutral.")

        # 4. Check shoulder positioning (prevent protraction or collapse)
        shoulder_elbow_angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
        shoulder_elbow_angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if shoulder_elbow_angle_left < 80 or shoulder_elbow_angle_right < 80:
            feedback.append("Ensure your shoulders are directly above your elbows/wrists.")

        # 5. Check if wrists are aligned under shoulders (for wrist support)
        wrist_under_shoulder_left = abs(left_shoulder.x - left_wrist.x)
        wrist_under_shoulder_right = abs(right_shoulder.x - right_wrist.x)

        if wrist_under_shoulder_left > 0.05 or wrist_under_shoulder_right > 0.05:
            feedback.append("Align your wrists directly under your shoulders for better support.")

    elif pose_name == "Warrior2":
        # Feedback for Warrior II pose
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # 1. Check if arms are parallel to the ground
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if left_arm_angle < 170:
            feedback.append("Raise your left arm higher and straighten it.")
        if right_arm_angle < 170:
            feedback.append("Raise your right arm higher and straighten it.")

        # 2. Check if the front knee is properly bent (~90 degrees)
        front_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if front_knee_angle > 95:
            feedback.append("Bend your front knee more to reach a 90-degree angle.")
        elif front_knee_angle < 85:
            feedback.append("Do not over-bend your front knee; keep it at 90 degrees.")

        # 3. Check if front knee is aligned over ankle
        if left_knee.x < left_ankle.x:
            feedback.append("Align your front knee over your ankle to avoid strain.")

        # 4. Check if back leg is straight
        back_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
        if back_leg_angle < 175:
            feedback.append("Straighten your back leg.")

        # 5. Check if hips are level
        hip_alignment = abs(left_hip.y - right_hip.y)
        if hip_alignment > 0.05:  # Adjust threshold as needed
            feedback.append("Keep your hips level.")

        # 6. Check if shoulders are level
        shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_alignment > 0.05:
            feedback.append("Keep your shoulders level.")

        # 7. Check if torso is upright
        torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        if torso_angle < 85 or torso_angle > 95:
            feedback.append("Keep your torso upright and avoid leaning forward or backward.")

        # 8. Check foot placement (front foot forward, back foot slightly turned)
        if abs(left_ankle.x - right_ankle.x) < 0.2:  # Adjust threshold
            feedback.append("Increase the distance between your feet for better stability.")

    elif pose_name == "Goddess":
        # Feedback for Goddess Pose
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        # 1. Check if knees are bent at approximately 90 degrees
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        if left_knee_angle > 100 or left_knee_angle < 80:
            feedback.append("Adjust your left knee to a 90-degree bend.")
        if right_knee_angle > 100 or right_knee_angle < 80:
            feedback.append("Adjust your right knee to a 90-degree bend.")

        # 2. Check if feet are wide apart
        feet_distance = abs(left_ankle.x - right_ankle.x)
        hip_distance = abs(left_hip.x - right_hip.x)
        if feet_distance < 1.5 * hip_distance:  # Ensuring feet are wider than hip width
            feedback.append("Widen your stance for better stability.")

        # 3. Check if torso is upright
        torso_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        if torso_angle < 85 or torso_angle > 95:
            feedback.append("Keep your torso upright and avoid leaning forward or backward.")

        # 4. Check if shoulders are level
        shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
        if shoulder_alignment > 0.05:
            feedback.append("Keep your shoulders level.")

        # 5. Check if arms are bent at 90 degrees
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if left_arm_angle < 85 or left_arm_angle > 95:
            feedback.append("Adjust your left arm to a 90-degree bend.")
        if right_arm_angle < 85 or right_arm_angle > 95:
            feedback.append("Adjust your right arm to a 90-degree bend.")

        # 6. Check if elbows are aligned with shoulders
        elbow_alignment_left = abs(left_shoulder.y - left_elbow.y)
        elbow_alignment_right = abs(right_shoulder.y - right_elbow.y)

        if elbow_alignment_left > 0.05:
            feedback.append("Lift your left elbow to align with your shoulder.")
        if elbow_alignment_right > 0.05:
            feedback.append("Lift your right elbow to align with your shoulder.")

    return feedback

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

        # Process the frame
        results = pose.process(image_rgb)

        # Initialize pose_name with a default value
        pose_name = "Unknown"

        if results.pose_landmarks:
            # Extract keypoints
            keypoints = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]

            # Classify the pose
            pose_id = classify_pose(keypoints, model)
            pose_name = pose_labels[pose_id]

            # Provide feedback based on the pose
            feedback = provide_feedback(results.pose_landmarks.landmark, pose_name)

            # Display the pose name in black color
            cv2.putText(frame, f"Pose: {pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Display feedback in black color
            y_offset = 60  # Start displaying feedback below the pose name
            for feedback_line in feedback:
                cv2.putText(frame, feedback_line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                y_offset += 30  # Adjust the y_offset to prevent overlap

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Yoga Pose Detection and Feedback', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()