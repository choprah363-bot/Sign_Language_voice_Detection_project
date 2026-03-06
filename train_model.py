import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

# =============================
# Load Valid Classes
# =============================
classes = []

for folder in sorted(os.listdir("data")):
    folder_path = os.path.join("data", folder)
    if os.path.isdir(folder_path) and len(os.listdir(folder_path)) > 0:
        classes.append(folder)

print("Valid Classes:", classes)

label_map = {label: idx for idx, label in enumerate(classes)}
print("Label Map:", label_map)

# =============================
# Load Dataset
# =============================
X = []
y = []

for label in classes:
    folder_path = os.path.join("data", label)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)
        X.append(data)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))

# =============================
# Train/Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# Build Model
# =============================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =============================
# Train
# =============================
model.fit(
    X_train, y_train,
    epochs=40,
    validation_data=(X_test, y_test),
    batch_size=32
)

# =============================
# Save Model
# =============================
model.save("gesture_model.keras")

# =============================
# Save Centroids (VERY IMPORTANT)
# =============================
centroids = {}

for class_name in classes:
    class_data = []
    folder_path = os.path.join("data", class_name)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        data_point = np.load(file_path)
        class_data.append(data_point)

    centroids[class_name] = np.mean(class_data, axis=0)

with open("centroids.pkl", "wb") as f:
    pickle.dump(centroids, f)

print("Model + Centroids Saved ✅")