# 🌿 Plant Disease Classifier – CNN + FastAPI

> **Live Demo**: Try it here 👉 [🔗 Plant Disease Predictor](https://potato-disease-classifier-frontend.onrender.com)  
> _(Note: The backend is hosted on **Render**, so the first request may take 1–2 minutes to respond while the server wakes up.)_

---

## 📌 Project Overview

This project is an end-to-end deep learning pipeline that classifies potato plant leaves into three categories:
- **Early Blight**
- **Late Blight**
- **Healthy**

It includes:
- A **Convolutional Neural Network (CNN)** model trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village?resource=download).
- A **FastAPI**-based backend for making predictions using base64-encoded images.
---

## 🧠 Model Architecture & Training

The model was developed and trained in a **Jupyter Notebook** using TensorFlow/Keras:

- **Image Size**: 256×256 (RGB)
- **Model Type**: Sequential CNN with convolutional blocks + dense layers
- **Data Augmentation**: Applied random flip and rotation
- **Training Epochs**: 50
- **Evaluation**: Visualized training/validation accuracy and loss, predicted sample images

After training, the model was saved using `model.save('1.keras')` for deployment.

---

## 🚀 FastAPI Backend

The backend is built using **FastAPI** and serves a single `/predict` endpoint.

### ✅ Features:
- Accepts base64-encoded images as input.
- Decodes, resizes, and preprocesses the image.
- Returns predicted class (`Early Blight`, `Late Blight`, or `Healthy`) and confidence score.

### 🔧 Example Request:
```json
POST /predict
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSk..."
}
```

### 🔁 Response:
```json
{
  "class": "Early Blight",
  "confidence": 0.9824
}
```

---

## 📂 Dataset

- **Source**: [Kaggle - PlantVillage](https://www.kaggle.com/datasets/arjuntejaswi/plant-village?resource=download)
- **Classes Used**: Potato – Early Blight, Late Blight, Healthy

---

## 🛠️ Technologies Used

| Area           | Tool / Framework       |
|----------------|------------------------|
| Model Training | Python, TensorFlow, Keras |
| API Backend    | FastAPI, Uvicorn       |
| Deployment     | Render.com             |
| Data Handling  | NumPy, PIL             |
| Communication  | JSON (base64 image transfer) |

---

## 🎓 Course Context

This project is part of a **Deep Learning course** offered by **CodeBasics**. The course provides comprehensive knowledge on the fundamentals of machine learning, with hands-on experience in building real-world applications. This project helped solidify core concepts such as model development, data preprocessing, and deployment, offering practical insights into the entire machine learning pipeline.

---

