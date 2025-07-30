from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import cohere
import pyttsx3
import cv2
import uuid
import traceback

# === Constants ===
IMG_SIZE = 224
MODEL_PATH = "indian_food_classifier_mobilenetv2acc97.h5"
CLASS_DIR = os.path.join("archive", "Indian Food Images")
AUDIO_PATH = "recipe_audio.mp3"

# === Load model ===
def load_food_model():
    return load_model(MODEL_PATH)

model = load_food_model()

# === Load class names ===
def get_class_names():
    return sorted(os.listdir(CLASS_DIR))

class_names = get_class_names()

# === Initialize Cohere ===
co = cohere.Client("TEhAeepxN3isXBsZ4hqBWM1aVt8rD97i9RJFwYTH")  # Replace with your key

# === Preprocess using OpenCV ===
def preprocess_image_opencv(image_pil):
    image_np = np.array(image_pil)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image_cv, (IMG_SIZE, IMG_SIZE))
    image_resized = image_resized.astype(np.float32) / 255.0
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(image_resized, axis=0)
    return img_array

# === Prediction ===
def predict_image(pil_img):
    img_array = preprocess_image_opencv(pil_img)
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    return class_names[pred_index]

# === Recipe Generation ===
def get_recipe_steps(recipe_name):
    response = co.chat(
        model="command-r-plus",
        message=f"Provide a detailed, step-by-step recipe for '{recipe_name}', including all ingredients and instructions.",
        temperature=0.7
    )
    return response.text.strip()

# === Text-to-speech ===
def text_to_speech(text):
    filename = f"{uuid.uuid4().hex}.mp3"
    path = f"static/{filename}"
    engine = pyttsx3.init()
    engine.save_to_file(text, path)
    engine.runAndWait()
    return f"/static/{filename}"

# === FastAPI App ===
app = FastAPI()

# === Allow frontend to access backend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Serve static audio ===
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Prediction API ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_pil = Image.open(file.file).convert("RGB")
        prediction = predict_image(image_pil)
        recipe_text = get_recipe_steps(prediction)
        audio_file = text_to_speech(recipe_text)

        return {
            "prediction": prediction,
            "recipe": recipe_text,
            "audio_url": f"/static/{os.path.basename(audio_file)}"
        }

    except Exception as e:
        print("=== ERROR TRACEBACK ===")
        traceback.print_exc()
        print("=== END TRACEBACK ===")
        return JSONResponse(status_code=500, content={"error": str(e)})
