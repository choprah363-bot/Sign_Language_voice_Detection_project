import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import pyttsx3
import threading

# =============================
# LOAD MODEL
# =============================
model = tf.keras.models.load_model("gesture_model.keras")

# =============================
# LOAD CLASSES
# =============================
classes = sorted([
    folder for folder in os.listdir("data")
    if os.path.isdir(os.path.join("data", folder))
])

print("Loaded Classes:", classes)

# =============================
# 🔥 ANGRY VOICE FUNCTION
# =============================
def speak(text):
    engine = pyttsx3.init()

    engine.setProperty('rate', 170)
    engine.setProperty('volume', 1.0)

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)

    text = text.replace("_", " ")

    if text == "Unknown":
        text = "I don't understand"

    text = text.upper() + " !!"

    engine.say(text)
    engine.runAndWait()
    engine.stop()

# =============================
# MEDIAPIPE
# =============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)

# =============================
# VARIABLES
# =============================
last_spoken = ""
current_label = ""
stable_count = 0

# =============================
# MAIN LOOP
# =============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # =============================
    # 🌙 AUTO LIGHT DETECTION
    # =============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    # dark vs bright handling
    if brightness < 60:
        frame = cv2.convertScaleAbs(frame, alpha=2.2, beta=80)
    else:
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)

    # =============================
    # 🔥 CLAHE (LOW LIGHT BOOST)
    # =============================
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    frame = cv2.merge((l,a,b))
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

    # =============================
    # NORMAL PROCESS
    # =============================
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    label = "No Hand"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)
            landmarks = landmarks - landmarks[0]
            landmarks = landmarks.flatten()

            input_data = np.array([landmarks])

            probs = model.predict(input_data, verbose=0)[0]

            class_id = np.argmax(probs)
            confidence = probs[class_id]

            sorted_probs = np.sort(probs)
            second_best = sorted_probs[-2]

            predicted = classes[class_id]

            # 🔥 improved unknown logic
            if confidence < 0.85 or (confidence - second_best) < 0.25:
                predicted = "Unknown"

            label = predicted

            print(f"{label} | {confidence:.2f}")

    # =============================
    # STABILITY
    # =============================
    if label == current_label:
        stable_count += 1
    else:
        current_label = label
        stable_count = 0

    # =============================
    # SPEAK
    # =============================
    if stable_count >= 8 and label != last_spoken:
        threading.Thread(target=speak, args=(label,)).start()
        last_spoken = label

    # =============================
    # DISPLAY
    # =============================
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    cv2.imshow(" Day/Night Angry Sign AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
