# import cv2
# import numpy as np
# import mediapipe as mp
# import winsound  # For beep alert on Windows
# from tensorflow.keras.models import load_model
# import streamlit as st
# from PIL import Image
# import tempfile

# # === Streamlit Title ===
# st.title("Real-time Drowsiness Detection")
# st.markdown("This app uses your webcam and a trained CNN model to detect if you're drowsy.")

# # === Load your trained model ===
# model = load_model(r'C:\Users\Karim Sherif\Desktop\WORKSPACE01\khlas_da_will_yshtghl\best_model.h5')

# # === MediaPipe for eye detection ===
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# def crop_eye_region(frame, landmarks, eye_idx):
#     h, w = frame.shape[:2]
#     points = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_idx])
#     x, y, w_, h_ = cv2.boundingRect(points)
#     eye = frame[y:y+h_, x:x+w_]
#     return eye

# def preprocess_eye(eye_img):
#     eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
#     eye_resized = cv2.resize(eye_gray, (224, 224))
#     eye_normalized = eye_resized / 255.0
#     return np.expand_dims(eye_normalized, axis=(0, -1))  # Shape: (1, 224, 224, 1)

# run = st.checkbox('Start Webcam')
# FRAME_WINDOW = st.image([])

# if run:
#     cap = cv2.VideoCapture(0)
#     st.warning("Press 'q' in webcam window to stop.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_frame)

#         status = "Face Not Detected"

#         if results.multi_face_landmarks:
#             landmarks = results.multi_face_landmarks[0].landmark

#             left_eye_img = crop_eye_region(frame, landmarks, LEFT_EYE_IDX)
#             right_eye_img = crop_eye_region(frame, landmarks, RIGHT_EYE_IDX)

#             try:
#                 left_input = preprocess_eye(left_eye_img)
#                 right_input = preprocess_eye(right_eye_img)

#                 left_pred = model.predict(left_input, verbose=0)[0][0]
#                 right_pred = model.predict(right_input, verbose=0)[0][0]

#                 left_label = "Open" if left_pred < 0.5 else "Closed"
#                 right_label = "Open" if right_pred < 0.5 else "Closed"

#                 if left_label == "Closed" and right_label == "Closed":
#                     status = "Drowsy"
#                     color = (0, 0, 255)
#                     winsound.Beep(1000, 500)
#                 else:
#                     status = "Alert"
#                     color = (0, 255, 0)

#                 cv2.putText(frame, f"L: {left_label} R: {right_label}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#             except Exception as e:
#                 print("Error processing eyes:", e)
#                 status = "Processing Error"
#                 color = (255, 0, 0)
#         else:
#             color = (0, 255, 255)

#         cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(frame)

#         # Handle stop
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()







import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import tempfile
import base64

# === Load your trained model ===
model = load_model("best_model.h5")

# === MediaPipe setup ===
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

def beep_html():
    beep_sound = """
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(beep_sound, unsafe_allow_html=True)

# === Streamlit UI ===
st.title("Drowsiness Detection App")
st.markdown("Upload a video or use webcam (locally only). On Streamlit Cloud, upload only.")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
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

                left_label = "Open" if left_pred < 0.5 else "Closed"
                right_label = "Open" if right_pred < 0.5 else "Closed"

                if left_label == "Closed" and right_label == "Closed":
                    status = "Drowsy"
                    color = (0, 0, 255)
                    beep_html()  # Trigger HTML beep
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
        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()