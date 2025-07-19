# main.py

# uvicorn main:app --host 0.0.0.0 --port 8000

import base64
from io import BytesIO
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Plant‑Disease Predictor (base‑64)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./models/1.keras")
INPUT_SIZE = (256, 256)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

class ImageData(BaseModel):
    image_base64: str


def decode_image(data_uri: str) -> np.ndarray:
    data_uri = data_uri.split(",", 1)[1]

    img_bytes = base64.b64decode(data_uri)

    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    img = img.resize(INPUT_SIZE)

    return np.array(img)



@app.post("/predict", summary="Predict plant disease from base‑64 image")
async def predict(data: ImageData):
    img_np = decode_image(data.image_base64)

    preds = MODEL.predict(np.expand_dims(img_np, 0))
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))

    return {"class": CLASS_NAMES[idx], "confidence": conf}