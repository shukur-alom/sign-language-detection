import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
classNames = ["Okay", "Peace", "Thumbs up", "Thumbs down",
              "Call me", "Stop", "I Love You", "Hello", "No", "Smile"]


FRAME_WINDOW = st.image([])
t = st.empty()

# Initialize the webcam
for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        break


while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            if prediction[0][classID]*100 >= 70.00:
                className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (20, 60), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 194, 247), 2, cv2.LINE_AA)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t.markdown(f"Predicted Gesture: {className}")
    
    if cv2.waitKey(1) == ord('q'):
        break
