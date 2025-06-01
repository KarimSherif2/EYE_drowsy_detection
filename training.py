#In[1]
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


#In[2]
train_dir = 'Drowsiness_Dataset_Grayscale/train'
val_dir = 'Drowsiness_Dataset_Grayscale/val'


# %%

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only normalization for validation
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Optional: check the class indices
print("Class indices:", train_generator.class_indices)




#MODEL TRAINING
# %%
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])


# %%
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# Save the final model
model.save('drowsiness_detector_final.h5')

print("✅ Training complete. Model saved.")

#visualization el model 
# %%
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()





# %%
import cv2
import numpy as np
import mediapipe as mp
import winsound  # For beep alert on Windows
from tensorflow.keras.models import load_model

# === Load your trained model ===
model = load_model(r'C:\Users\Karim Sherif\Desktop\WORKSPACE01\khlas_da_will_yshtghl\best_model.h5')

# === MediaPipe for eye detection ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

def crop_eye_region(frame, landmarks, eye_idx):
    h, w = frame.shape[:2]
    points = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_idx])
    x, y, w_, h_ = cv2.boundingRect(points)
    eye = frame[y:y+h_, x:x+w_]
    return eye

def preprocess_eye(eye_img):
    eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_resized = cv2.resize(eye_gray, (224, 224))
    eye_normalized = eye_resized / 255.0
    return np.expand_dims(eye_normalized, axis=(0, -1))  # Shape: (1, 224, 224, 1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    status = "Face Not Detected"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye_img = crop_eye_region(frame, landmarks, LEFT_EYE_IDX)
        right_eye_img = crop_eye_region(frame, landmarks, RIGHT_EYE_IDX)

        try:
            left_input = preprocess_eye(left_eye_img)
            right_input = preprocess_eye(right_eye_img)

            left_pred = model.predict(left_input, verbose=0)[0][0]
            right_pred = model.predict(right_input, verbose=0)[0][0]

            # Fix: Adjust based on your model's logic (inverted logic fix)
            left_label = "Open" if left_pred < 0.5 else "Closed"
            right_label = "Open" if right_pred < 0.5 else "Closed"

            if left_label == "Closed" and right_label == "Closed":
                status = "Drowsy"
                color = (0, 0, 255)
                # === Beep sound alert ===
                winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
            else:
                status = "Alert"
                color = (0, 255, 0)

            cv2.putText(frame, f"L: {left_label} R: {right_label}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            print("Error processing eyes:", e)
            status = "Processing Error"
            color = (255, 0, 0)
    else:
        color = (0, 255, 255)

    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



#In[]
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your trained model
model = load_model('drowsiness_detector_final.h5')

# Define image parameters
IMG_SIZE = 224
BATCH_SIZE = 32

# Create validation generator
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    'path_to_val_dir',  # Replace with your validation directory path
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',  # Match your training input
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

# %%
