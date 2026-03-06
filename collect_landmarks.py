import cv2
import mediapipe as mp
import numpy as np
import os

gesture_name = "unknown"  # हर बार बदलो: hello / i_love_you / yes

save_path = f"data/{gesture_name}"
os.makedirs(save_path, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,


    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

count = 0

while count < 300:  # कम से कम 300 samples लो

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

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

            # normalize (बहुत जरूरी)
            landmarks = landmarks - landmarks[0]

            landmarks = landmarks.flatten()

            np.save(f"{save_path}/{count}.npy", landmarks)

            count += 1
            print("Saved:", count)

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()