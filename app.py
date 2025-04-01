from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()
model = YOLO("best_model.pt")

@app.post("/predict/")
async def predict(file: UploadFile):
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img)
    return {"detections": str(results)}

