import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Load trained model
model = tf.keras.models.load_model("gesture_model.keras")

# Load class names
classes = sorted([
    folder for folder in os.listdir("data")
    if os.path.isdir(os.path.join("data", folder))
])

print("Loaded Classes:", classes)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
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

            # normalize
            landmarks = landmarks - landmarks[0]
            landmarks = landmarks.flatten()

            input_data = np.array([landmarks])

            probs = model.predict(input_data, verbose=0)[0]

            # highest probability
            class_id = np.argmax(probs)
            confidence = probs[class_id]

            # second best probability
            sorted_probs = np.sort(probs)
            second_best = sorted_probs[-2]

            predicted = classes[class_id]

            # UNKNOWN RULE
            if confidence < 0.80 or (confidence - second_best) < 0.30:
                predicted = "Unknown"

            label = predicted

            print("confidence:", confidence, "gap:", confidence-second_best)

    cv2.putText(
        frame,
        label,
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()