import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import os

engine = pyttsx3.init()

model = tf.keras.models.load_model("sign_model.keras")
gestures = os.listdir("data")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks)
            gesture = gestures[np.argmax(prediction)]

            cv2.putText(frame, gesture, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            engine.say(gesture)
            engine.runAndWait()

    cv2.imshow("Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()