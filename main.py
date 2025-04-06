from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import mediapipe as mp
import joblib
from io import BytesIO

app = FastAPI()

# Load model and label encoder
model = joblib.load("hand_gesture_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

@app.post("/predict/")
async def predict_hand_gesture(file: UploadFile = File(...)):
    # Read image from uploaded file
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return JSONResponse({"error": "No hand detected"}, status_code=400)

    # Process the first hand detected
    landmarks = []
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    if len(landmarks) != 63:
        return JSONResponse({"error": "Incomplete hand landmarks"}, status_code=400)

    # Predict gesture
    X_input = np.array(landmarks).reshape(1, -1)
    y_pred = model.predict(X_input)
    gesture = label_encoder.inverse_transform(y_pred)[0]

    return {"gesture": gesture}
