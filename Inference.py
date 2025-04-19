# Import necessary libraries
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

# --- Constants ---
CNN_WEIGHTS_PATH = "cnn_yoga_trained.weights.h5" # saved CNN weights file
CNN_INPUT_SHAPE = (75, 75, 3) # Should match the target_size in your notebook
NUM_CLASSES = 5 # number of yoga poses

# This order MUST match the indices assigned by ImageDataGenerator
# Check train_generator.class_indices in your notebook
CLASS_LABELS = [
    'downdog',
    'goddess',
    'plank',
    'tree',
    'warrior2'
] 

# --- Landmark Indices (MediaPipe Standard) ---
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

# --- Helper Functions ---

def inFrame(pose_landmarks):
    """ Checks if essential landmarks are visible. Adjust as needed. """
    if pose_landmarks is None:
        return False
    landmarks = pose_landmarks.landmark
    threshold = 0.5
    required = [
        LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.RIGHT_SHOULDER,
        LandmarkIndices.LEFT_HIP, LandmarkIndices.RIGHT_HIP,
        LandmarkIndices.LEFT_KNEE, LandmarkIndices.RIGHT_KNEE,
        LandmarkIndices.LEFT_ANKLE, LandmarkIndices.RIGHT_ANKLE
    ]
    return all(landmarks[i].visibility > threshold for i in required if i < len(landmarks))

def calculate_angle(a, b, c):
    """
    Compute the angle (in degrees) at point b formed by points a-b-c.
    
    Returns None for invalid input or degenerate configurations.
    """
    try:
        pa = np.array([a.x, a.y])
        pb = np.array([b.x, b.y])
        pc = np.array([c.x, c.y])
    except AttributeError:
        return None

    vec1 = pa - pb
    vec2 = pc - pb
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return None
    
    cos_theta = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# --- Define CNN Model Architecture ---
def create_cnn_model(input_shape, num_classes):
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
        layers.Dense(num_classes, activation='softmax') # Use num_classes here
    ])
    return model

# --- Default Angle Ranges ---
POSE_ANGLE_RULES = {
    'tree': { # Assuming 'tree' is one of your class labels
        'standing_knee':   ((LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE, LandmarkIndices.LEFT_ANKLE), 165, 180),
        'raised_knee':     ((LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_ANKLE),  60, 140),
        'hip_abduction':   ((LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_HIP, LandmarkIndices.LEFT_HIP),   40,  90),
    },
    'warrior2': { # Assuming 'warrior2' is one of your class labels
        'front_knee':      ((LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE, LandmarkIndices.LEFT_ANKLE),  80, 110),
        'back_knee':       ((LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_ANKLE),165, 180),
        'left_elbow':      ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_ELBOW, LandmarkIndices.LEFT_WRIST),160,180),
        'right_elbow':     ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_ELBOW, LandmarkIndices.RIGHT_WRIST),160,180),
    },
    'downdog': { # Add rules for other poses if needed
         'left_knee':       ((LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE, LandmarkIndices.LEFT_ANKLE),  160, 180),
         'right_knee':      ((LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_ANKLE), 160, 180),
         'left_elbow':      ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_ELBOW, LandmarkIndices.LEFT_WRIST),160,180),
         'right_elbow':     ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_ELBOW, LandmarkIndices.RIGHT_WRIST),160,180),
         'hip_angle':       ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE),   70, 110),
    },
    'plank': {
         'body_line_left':  ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_ANKLE), 170, 185),
         'body_line_right': ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_ANKLE), 170, 185),
         'left_elbow':      ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_ELBOW, LandmarkIndices.LEFT_WRIST),160,185),
         'right_elbow':     ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_ELBOW, LandmarkIndices.RIGHT_WRIST),160,185),
    },
    'goddess': {
         'left_knee':       ((LandmarkIndices.LEFT_HIP, LandmarkIndices.LEFT_KNEE, LandmarkIndices.LEFT_ANKLE),  80, 110),
         'right_knee':      ((LandmarkIndices.RIGHT_HIP, LandmarkIndices.RIGHT_KNEE, LandmarkIndices.RIGHT_ANKLE), 80, 110),
         'left_elbow':      ((LandmarkIndices.LEFT_SHOULDER, LandmarkIndices.LEFT_ELBOW, LandmarkIndices.LEFT_WRIST), 80, 100),
         'right_elbow':     ((LandmarkIndices.RIGHT_SHOULDER, LandmarkIndices.RIGHT_ELBOW, LandmarkIndices.RIGHT_WRIST), 80, 100),
    },
    # Add rules for ALL your classes if you want angle feedback
    'Unknown Pose': {} # Generic label for low confidence or no pose
}


# --- Load CNN Model and Labels ---
try:
    # 1. Create the model architecture
    model = create_cnn_model(CNN_INPUT_SHAPE, NUM_CLASSES)
    # 2. Load the saved weights
    model.load_weights(CNN_WEIGHTS_PATH)
    print(f"CNN Model weights '{CNN_WEIGHTS_PATH}' loaded successfully.")

    # Use the predefined CLASS_LABELS
    labels = CLASS_LABELS
    if len(labels) != NUM_CLASSES:
        raise ValueError(f"Number of classes in CLASS_LABELS ({len(labels)}) does not match NUM_CLASSES ({NUM_CLASSES})")

    # Validate POSE_ANGLE_RULES keys against loaded labels
    rule_keys = set(POSE_ANGLE_RULES.keys())
    loaded_labels_set = set(labels)
    missing_rules = loaded_labels_set - rule_keys - {'Unknown Pose'}
    extra_rules = rule_keys - loaded_labels_set - {'Unknown Pose'}
    print(f"Loaded Labels for CNN: {labels}")
    if missing_rules: print(f"WARNING: Labels missing from POSE_ANGLE_RULES: {missing_rules}")
    if extra_rules: print(f"WARNING: POSE_ANGLE_RULES has keys not in loaded labels: {extra_rules}")

    valid_poses = [p for p in labels if p != 'Unknown Pose'] # For suggestions
    print(f"Trainable Poses for Suggestions/Feedback: {valid_poses}")

except FileNotFoundError:
    print(f"ERROR: Could not find CNN weights file '{CNN_WEIGHTS_PATH}'.")
    exit()
except Exception as e:
    print(f"Error loading CNN model or weights: {e}")
    exit()

# --- MediaPipe Pose Setup ---
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # For drawing specs

# --- LLM Setup ---
llm_model = None
llm_available = False
try:
    api_key = "AIzaSyDms_HimOJuI0Gs69vxCtiziDZAbKgWZYU"
    if not api_key or api_key == "YOUR_API_KEY_HERE":
       raise ValueError("API Key not found or not replaced.")

    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Google Generative AI Model (Gemini) Loaded Successfully.")
    llm_available = True
except Exception as e:
    print("-" * 50)
    print(f"LLM Setup Warning: {e}")
    print("LLM-based feedback will be disabled.")
    print("-" * 50)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- State Variables ---
current_pose_label = None
predicted_label = None
confidence = 0.0
last_feedback_pose = None
pose_start_time = None
idle_start_time = None
suggestion_given = False
feedback_generated_for_current_hold = False
feedback_text = "Initializing..."
POSE_HOLD_DURATION_THRESHOLD = 1
SUGGESTION_IDLE_THRESHOLD = 5.0
CONFIDENCE_THRESHOLD = 0.50

# --- Main Loop ---
while True:
    ret, frm = cap.read()
    if not ret or frm is None:
        print("Error reading frame. Skipping.")
        time.sleep(0.5)
        continue

    # Create UI window, flip frame
    window = np.zeros((940, 940, 3), dtype="uint8")
    frm = cv2.flip(frm, 1)

    # --- MediaPipe Pose Estimation (Run ONCE per frame) ---
    frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    frm_rgb.flags.writeable = False
    pose_results = pose_detector.process(frm_rgb)
    frm_rgb.flags.writeable = True
    display_frm = frm
    # --- Initialize state for this frame ---
    is_idle = True
    pose_analysis_possible = False
    landmarks = None
    # --- CNN Prediction Path ---
    if pose_results.pose_landmarks: # Only predict if landmarks are detected
        landmarks = pose_results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(
            display_frm,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        if inFrame(pose_results.pose_landmarks):
            pose_analysis_possible = True
            lst = []
            angle_feedback_points = []
            correct_angles = []
            incorrect_angles = []
            # Preprocess frame for CNN
            img_for_cnn = cv2.resize(frm, (CNN_INPUT_SHAPE[1], CNN_INPUT_SHAPE[0])) # Resize (width, height)
            img_for_cnn = img_for_cnn / 255.0 # Rescale
            img_for_cnn = np.expand_dims(img_for_cnn, axis=0) # Add batch dimension

            # Predict using CNN
            try:
                predictions = model.predict(img_for_cnn, verbose=0)
                confidence = np.max(predictions[0])
                predicted_idx = np.argmax(predictions[0])
                predicted_label = labels[predicted_idx]
            except Exception as e: 
                print(f"Model prediction error: {e}")
                predicted_label = "Error"
                confidence = 0.0
            # Determine predicted label based on confidence threshold
            if confidence >= CONFIDENCE_THRESHOLD and predicted_label != 'nothing' and predicted_label != "Error":
                is_idle = False # Not idle if confident prediction
                idle_start_time = None
                suggestion_given = False
                if predicted_label == current_pose_label:
                    # Pose held consistently
                    if pose_start_time is None:
                        pose_start_time = time.time()

                    elif (time.time() - pose_start_time >= POSE_HOLD_DURATION_THRESHOLD) and not feedback_generated_for_current_hold:
                        feedback_generated_for_current_hold = True
                        last_feedback_pose = predicted_label

                        # --- Detailed Angle Analysis ---
                        if predicted_label in POSE_ANGLE_RULES:
                            rules = POSE_ANGLE_RULES[predicted_label]
                            # Simplified check - assumes rules cover main angles
                            for angle_name, (lm_indices, min_angle, max_angle) in rules.items():
                                # Need to handle poses with left/right variants more robustly if rules are specific
                                lm1, lm2, lm3 = landmarks[lm_indices[0]], landmarks[lm_indices[1]], landmarks[lm_indices[2]]
                                if lm1.visibility > 0.5 and lm2.visibility > 0.5 and lm3.visibility > 0.5:
                                    current_angle = calculate_angle(lm1, lm2, lm3)
                                    if current_angle is not None:
                                        if min_angle <= current_angle <= max_angle:
                                            correct_angles.append(angle_name.replace('_', ' '))
                                        else:
                                            direction = "Increase" if current_angle < min_angle else "Decrease"
                                            incorrect_angles.append(f"{angle_name.replace('_', ' ')} ({direction} needed)")
                                else:
                                     pass # Landmark low visibility

                        # --- Generate Detailed Feedback ---
                        feedback_prompt = ""
                        fallback_feedback = ""
                        rules = POSE_ANGLE_RULES.get(predicted_label, {})
                        angle_values = {}
                        for name,(inds,mn,mx) in rules.items():
                            a,b,c = landmarks[inds[0]], landmarks[inds[1]], landmarks[inds[2]]
                            if a.visibility>0.5 and b.visibility>0.5 and c.visibility>0.5:
                                ang = calculate_angle(a,b,c)
                                angle_values[name] = (ang,mn,mx)

                        if not correct_angles and not incorrect_angles:
                             fallback_feedback = f"{predicted_label}: Looking good! Maintain focus."
                        else:
                            corr_str = f"Correct: {', '.join(correct_angles) if correct_angles else 'None identified'}."
                            incorr_str = f"Adjust: {', '.join(incorrect_angles) if incorrect_angles else 'None needed'}."
                            # Simple fallback
                            fallback_feedback = f"{predicted_label}: {corr_str} {incorr_str}"

                        if llm_available and llm_model:
                            try:
                                report = "\n".join(
                                f"{n.replace('_',' ')}: {v:.1f}° (target {mn}-{mx}°)"
                                for n,(v,mn,mx) in angle_values.items()
                                )
                                prompt = f"""You are a yoga coach. The user is holding {predicted_label} with these joint angles: {report}. For each joint, suggest increase, decrease, or maintain and by approx how many degrees. One line per joint."""

                                response = llm_model.generate_content(prompt)
                                feedback_text = response.text.strip().replace("*", "")
                            except Exception as e:
                                print(f"LLM detailed feedback failed: {e}")
                                feedback_text = fallback_feedback # Use fallback
                        else:
                            feedback_text = fallback_feedback # Use fallback if LLM unavailable

                else:
                    # New valid pose detected
                    current_pose_label = predicted_label
                    pose_start_time = time.time()
                    feedback_generated_for_current_hold = False
                    feedback_text = f"Detected: {current_pose_label}. Hold..."
            # --- End of valid pose check ---
            else:
                 # Confidence low or 'nothing' detected during analysis phase
                 is_idle = True # Treat as idle if analysis doesn't yield a good pose
                 current_pose_label = None
                 pose_start_time = None
                 feedback_generated_for_current_hold = False
                 if predicted_label != "Error":
                     feedback_text = "Pose unclear. Adjust position?"
        # --- End of inFrame check ---
        else:
            # Landmarks detected, but not fully in frame
            is_idle = True # Treat as idle if not fully visible
            current_pose_label = None
            pose_start_time = None
            feedback_generated_for_current_hold = False
            feedback_text = "Body detected. Ensure full body is visible."
    # --- End of landmark detection check ---
    else:
        # No landmarks detected at all
        is_idle = True
        current_pose_label = None
        pose_start_time = None
        feedback_generated_for_current_hold = False
        predicted_label = None
        confidence = 0.0
        feedback_text = "No body detected. Position yourself."
        cv2.putText(display_frm, "No body detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


    # --- Handle Idle State & Suggestions ---
    if is_idle:
        # Reset active pose tracking
        current_pose_label = None
        pose_start_time = None
        feedback_generated_for_current_hold = False # Important reset

        if idle_start_time is None:
            idle_start_time = time.time()
        elif (time.time() - idle_start_time >= SUGGESTION_IDLE_THRESHOLD) and not suggestion_given:
             if llm_available and llm_model and valid_poses:
                try:
                    suggested_pose = random.choice(valid_poses)
                    prompt = f"""
                    The user seems idle during a yoga practice session.
                    Briefly (1 sentence) and encouragingly suggest they try '{suggested_pose}'.
                    Make it sound natural and inviting.
                    Example: Perhaps try a grounding {suggested_pose} next?
                    Example: Feeling ready? {suggested_pose} is a great pose to try!
                    """
                    response = llm_model.generate_content(prompt)
                    feedback_text = response.text.strip().replace("*", "")
                    suggestion_given = True # Prevent immediate repetition
                except Exception as e:
                    print(f"LLM suggestion failed: {e}")
                    # Keep the last non-suggestion feedback_text
             else:
                 # Fallback if LLM unavailable or no valid poses
                 feedback_text = "Ready for the next pose?" # Simple fallback
                 suggestion_given = True # Prevent loop if LLM always fails


    # --- Display Info & Feedback ---
    # Display Prediction Info (even if waiting/N/A)
    cv2.putText(window, f"Pose: {predicted_label if pose_analysis_possible and predicted_label else 'N/A'}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(window, f"Correctness Percentage: {confidence*100:.2f}%" if pose_analysis_possible and predicted_label else "Correctness Percentage: N/A", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Display Feedback Text
    y0, dy = 150, 35
    max_chars_per_line = 55
    lines = []
    if feedback_text:
        # Split into lines respecting max_chars_per_line AND existing newlines
        for paragraph in feedback_text.split('\n'):
            words = paragraph.split(' ')
            current_line = ""
            for word in words:
                if not current_line: # First word on the line
                     current_line = word
                elif len(current_line) + len(word) + 1 <= max_chars_per_line:
                    current_line += " " + word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line: # Add the last line of the paragraph
                lines.append(current_line)

    # Determine text color
    feedback_color = (0, 255, 0) # Green default
    lower_feedback = feedback_text.lower() if feedback_text else ""
    if "adjust:" in lower_feedback or "try to" in lower_feedback or "focus on" in lower_feedback or "increase" in lower_feedback or "decrease" in lower_feedback or "ensure" in lower_feedback or "suggestion" in lower_feedback or "maybe try" in lower_feedback:
        feedback_color = (0, 200, 255) # Yellow/Orange
    elif "error" in lower_feedback or "unclear" in lower_feedback or "no body" in lower_feedback or "waiting" in lower_feedback or "n/a" in lower_feedback:
         feedback_color = (0, 0, 255) # Red

    for i, line in enumerate(lines[:6]): # Limit lines displayed
        y = y0 + i * dy
        cv2.putText(window, line, (100, y), cv2.FONT_ITALIC, 0.8, feedback_color, 2)



    # --- Combine SKELETON Video Feed into the Window ---
    target_h, target_w = 480, 640
    h_skel, w_skel, _ = display_frm.shape
    resized_skel = None

    if h_skel > 0 and w_skel > 0:
        scale = min(target_w / w_skel, target_h / h_skel)
        new_w, new_h = int(w_skel * scale), int(h_skel * scale)
        if new_w > 0 and new_h > 0:
             resized_skel = cv2.resize(display_frm, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if resized_skel is not None:
        area_start_x, area_start_y = 150, 400
        start_x = area_start_x + (target_w - resized_skel.shape[1]) // 2
        start_y = area_start_y + (target_h - resized_skel.shape[0]) // 2
        try:
            window[start_y : start_y + resized_skel.shape[0], start_x : start_x + resized_skel.shape[1]] = resized_skel
        except ValueError as e:
            print(f"Skeleton placement error: {e}")
            cv2.rectangle(window, (area_start_x, area_start_y), (area_start_x + target_w, area_start_y + target_h), (0,0,255), 2)
    else:
        cv2.rectangle(window, (150, 400), (150 + target_w, 400 + target_h), (0,0,255), 1)


    # --- Show the Final Window ---
    cv2.imshow("Yoga Pose Feedback System", window)

    # --- Exit Condition ---
    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break

# --- Cleanup ---
print("Exiting program...")
cap.release()
cv2.destroyAllWindows()
pose_detector.close()