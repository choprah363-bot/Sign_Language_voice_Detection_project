import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

data = []
labels = []
gestures = os.listdir("data")

for idx, gesture in enumerate(gestures):
    for file in os.listdir(f"data/{gesture}"):
        landmark = np.load(f"data/{gesture}/{file}")
        data.append(landmark)
        labels.append(idx)

X = np.array(data)
y = np.array(labels)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(gestures), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=50)

model.save("sign_model.keras")
print("Model Trained Successfully ✅")