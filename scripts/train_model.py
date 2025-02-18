import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from collections import Counter

# Load the dataset
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values  # Keypoints
    y = df.iloc[:, -1].values   # Labels

 # Downsample to balance classes (set all to 228 samples)
    target_samples = 228
    df_balanced = df.groupby('label').apply(lambda x: x.sample(n=target_samples, random_state=42)).reset_index(drop=True)

    # Verify the new distribution
    class_counts = df_balanced['label'].value_counts()
    print("\nBalanced Class Distribution:")
    print(class_counts)

    # Separate features (X) and labels (y)
    X = df_balanced.iloc[:, :-1].values  # Keypoints
    y = df_balanced.iloc[:, -1].values   # Labels

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder

    # Check class balance
    class_counts = Counter(y)
    print("Initial Class Distribution:")
    for label, count in class_counts.items():
        print(f"{label}: {count} samples")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder

# Build the model
def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),  # Input layer
        Dropout(0.2),  # Regularization
        Dense(64, activation='relu'),  # Hidden layer
        Dropout(0.2),  # Regularization
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function
if __name__ == "__main__":
    csv_file = "../data/yoga_poses_keypoints.csv"
    X, y, label_encoder = load_dataset(csv_file)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(X_train.shape[1], len(label_encoder.classes_))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    model.save("../models/yoga_pose_model.h5")
    print("\nModel saved to ../models/yoga_pose_model.h5")

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

   