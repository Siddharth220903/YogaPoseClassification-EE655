# Import Required Packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf  
from tensorflow.keras import layers, models
import time
import random
import google.generativeai as genai
import os
import math

# --- Configuration Constants ---
CNN_WEIGHTS_PATH = "cnn_yoga_trained.weights.h5"  # Path to saved CNN weights
CNN_INPUT_SHAPE = (75, 75, 3)  # Input image size for the CNN
NUM_CLASSES = 5  # Number of yoga pose classes

# Class labels must match the order used during training
CLASS_LABELS = [
    'downdog',
    'goddess',
    'plank',
    'tree',
    'warrior2'
]

# MediaPipe landmark indices for reference
class LandmarkIndices:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# Angle thresholds for each pose (min, max) based on landmark triples
POSE_ANGLE_RULES = {
    'tree': {
        'standing_knee':   ((LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE, LandmarkIndices.LEFT_ANKLE), 165, 180),
        'raised_knee':     ((LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_ANKLE), 60, 140),
        'hip_abduction':   ((LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_HIP, LandmarkIndices.LEFT_HIP),   40,  90),
    },
    'warrior2': {
        'front_knee':      ((LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE, LandmarkIndices.LEFT_ANKLE),  80, 110),
        'back_knee':       ((LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_ANKLE),165, 180),
        'left_elbow':      ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_ELBOW, LandmarkIndices.LEFT_WRIST),160,180),
        'right_elbow':     ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_ELBOW, LandmarkIndices.RIGHT_WRIST),160,180),
    },
    'downdog': {
        'left_knee':       ((LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE, LandmarkIndices.LEFT_ANKLE),  160, 180),
        'right_knee':      ((LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_ANKLE), 160, 180),
        'left_elbow':      ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_ELBOW, LandmarkIndices.LEFT_WRIST),160,180),
        'right_elbow':     ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_ELBOW, LandmarkIndices.RIGHT_WRIST),160,180),
        'hip_angle':       ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE),   70, 110),
    },
    'plank': {
        'body_line_left':  ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_ANKLE), 170, 185),
        'body_line_right': ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_ANKLE),170, 185),
        'left_elbow':      ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_ELBOW, LandmarkIndices.LEFT_WRIST),160,185),
        'right_elbow':     ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_ELBOW, LandmarkIndices.RIGHT_WRIST),160,185),
    },
    'goddess': {
        'left_knee':       ((LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE, LandmarkIndices.LEFT_ANKLE),  80, 110),
        'right_knee':      ((LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_ANKLE), 80, 110),
        'left_elbow':      ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_ELBOW, LandmarkIndices.LEFT_WRIST), 80, 100),
        'right_elbow':     ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_ELBOW, LandmarkIndices.RIGHT_WRIST), 80, 100),
    },
    'Unknown Pose': {}  # Fallback for unrecognized poses
}

# --- Helper Functions ---
def inFrame(pose_landmarks):
    """
    Returns True if all key lower-body and shoulder landmarks are visible above threshold.
    """
    if not pose_landmarks:
        return False
    landmarks = pose_landmarks.landmark
    threshold = 0.5
    required = [
        LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.RIGHT_SHOULDER,
        LandmarkIndices.LEFT_HIP, LandmarkIndices.RIGHT_HIP,
        LandmarkIndices.LEFT_KNEE, LandmarkIndices.RIGHT_KNEE,
        LandmarkIndices.LEFT_ANKLE, LandmarkIndices.RIGHT_ANKLE
    ]
    return all(landmarks[i].visibility > threshold for i in required)


def calculate_angle(a, b, c):
    """
    Calculates the angle in degrees at joint b given three landmarks a-b-c.
    Returns None for invalid or degenerate configurations.
    """
    try:
        pa = np.array([a.x, a.y])
        pb = np.array([b.x, b.y])
        pc = np.array([c.x, c.y])
    except AttributeError:
        return None

    v1 = pa - pb
    v2 = pc - pb
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None

    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# --- CNN Model Definition ---
def create_cnn_model(input_shape, num_classes):
    """
    Builds the convolutional neural network architecture matching the training setup.
    """
    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', padding='Same', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3,3), activation='relu', padding='Same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3,3), activation='relu', padding='Same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# --- Initialization: Load model and verify labels ---
try:
    model = create_cnn_model(CNN_INPUT_SHAPE, NUM_CLASSES)
    model.load_weights(CNN_WEIGHTS_PATH)
    print(f"Loaded CNN weights from '{CNN_WEIGHTS_PATH}'.")

    if len(CLASS_LABELS) != NUM_CLASSES:
        raise ValueError("Mismatch between NUM_CLASSES and CLASS_LABELS length.")

    loaded = set(CLASS_LABELS)
    rule_keys = set(POSE_ANGLE_RULES.keys())
    missing = loaded - rule_keys - {'Unknown Pose'}
    extra = rule_keys - loaded - {'Unknown Pose'}
    print(f"Classes: {CLASS_LABELS}")
    if missing:
        print(f"WARNING: Missing angle rules for: {missing}")
    if extra:
        print(f"WARNING: Extra rules for undefined poses: {extra}")

except Exception as e:
    print(f"Model loading failed: {e}")
    exit()

# --- MediaPipe pose estimator setup ---
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(model_complexity=1,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# --- Google Generative AI setup for feedback ---
llm_available = False
try:
    api_key = os.getenv("GENAI_API_KEY")
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    llm_available = True
    print("Google Gemini loaded for feedback generation.")
except Exception as e:
    print(f"LLM disabled: {e}")

# --- Webcam Initialization ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# --- Runtime variables ---
current_pose_label = None
confidence = 0.0
pose_start_time = None
idle_start_time = None
suggestion_given = False
feedback_generated_for_current_hold = False
feedback_text = "Initializing..."

# Thresholds
POSE_HOLD_DURATION_THRESHOLD = 1    # seconds to hold a pose before feedback
SUGGESTION_IDLE_THRESHOLD = 5.0     # seconds of inactivity before suggesting a new pose
CONFIDENCE_THRESHOLD = 0.50         # CNN confidence threshold

# --- Main loop: capture frame, estimate pose, classify, and provide feedback ---
while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.5)
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = pose_detector.process(img_rgb)
    img_rgb.flags.writeable = True

    display = frame.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            display,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

        # Only proceed if full body is in frame
        if inFrame(results.pose_landmarks):
            # Prepare image for classification
            resized = cv2.resize(frame, (CNN_INPUT_SHAPE[1], CNN_INPUT_SHAPE[0])) / 255.0
            test_img = np.expand_dims(resized, axis=0)
            preds = model.predict(test_img, verbose=0)[0]
            confidence = float(np.max(preds))
            idx = int(np.argmax(preds))
            predicted_label = CLASS_LABELS[idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                idle_start_time = None
                suggestion_given = False

                if predicted_label == current_pose_label:
                    # Pose held; check duration for feedback
                    if pose_start_time is None:
                        pose_start_time = time.time()
                    elif not feedback_generated_for_current_hold and (time.time() - pose_start_time >= POSE_HOLD_DURATION_THRESHOLD):
                        feedback_generated_for_current_hold = True
                        # Angle analysis and feedback generation (LLM or fallback)
                        # ... existing logic ...
                else:
                    # New pose detected
                    current_pose_label = predicted_label
                    pose_start_time = time.time()
                    feedback_generated_for_current_hold = False
                    feedback_text = f"Detected: {current_pose_label}. Hold..."
            else:
                # Low confidence
                current_pose_label = None
                pose_start_time = None
                feedback_text = "Pose unclear. Adjust position?"
        else:
            # Partial body in frame
            current_pose_label = None
            pose_start_time = None
            feedback_text = "Ensure full body is visible."
    else:
        # No person detected
        current_pose_label = None
        pose_start_time = None
        feedback_text = "No body detected. Position yourself."

    # Handle idle suggestions
    # ... existing suggestion logic ...

    # Overlay information and feedback on screen
    # ... drawing code ...

    cv2.imshow("Yoga Pose Feedback System", display)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pose_detector.close()
