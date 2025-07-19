# main.py

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model("./models/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    images = np.array(Image.open(BytesIO(data)))
    return images

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)

    predicted_class = np.argmax(prediction[0])
    predicted_percentage = np.max(prediction[0])

    return {
        "class": CLASS_NAMES[predicted_class],
        "confidence": float(predicted_percentage)
    }
