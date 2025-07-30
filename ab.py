import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import cohere
import pyttsx3
import cv2

# Constants
IMG_SIZE = 224
MODEL_PATH = r"C:\Users\HOME\Desktop\New folder\indian_food_classifier_mobilenetv2acc97.h5"
CLASS_DIR = "archive\Indian Food Images"
AUDIO_PATH = "recipe_audio.mp3"

# Load model
def load_food_model():
    return load_model(MODEL_PATH)

model = load_food_model()

# Load class names
def get_class_names():
    return sorted(os.listdir(CLASS_DIR))

class_names = get_class_names()

# Initialize Cohere
co = cohere.Client("TEhAeepxN3isXBsZ4hqBWM1aVt8rD97i9RJFwYTH")  # Replace with your actual key
def preprocess_image_opencv(image_pil):
    # Convert PIL to OpenCV format (numpy array)
    image_np = np.array(image_pil)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

    # Resize the image
    image_resized = cv2.resize(image_cv, (IMG_SIZE, IMG_SIZE))

    # Normalize and cast to float32
    image_resized = image_resized.astype(np.float32) / 255.0

    # Convert back to RGB (OpenCV expects BGR, but model expects RGB)
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Add batch dimension
    img_array = np.expand_dims(image_resized, axis=0)
    return img_array
def predict_image(pil_img):
    img_array = preprocess_image_opencv(pil_img)
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    return class_names[pred_index]

# Recipe generation
def get_recipe_steps(recipe_name):
    response = co.chat(
        model="command-r-plus",
        message=f"Provide a detailed, step-by-step recipe for '{recipe_name}', including all ingredients and instructions.",
        temperature=0.7
    )
    return response.text.strip()

# Text-to-speech
def text_to_speech(text, path=AUDIO_PATH):
    engine = pyttsx3.init()
    engine.save_to_file(text, path)
    engine.runAndWait()
    return path

# Streamlit UI Setup
st.set_page_config(page_title="🍛 Indian Food Detector + Recipe Audio", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #ff6347;'>🍽 Indian Food Detector</h1>
    <p style='text-align: center;'>Upload an image of Indian food and get a delicious recipe with audio instructions!</p>
""", unsafe_allow_html=True)

# Upload Image Section
with st.container():
    st.subheader("📤 Upload Food Image")
    uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="🍛 Uploaded Image", use_column_width=True)

# Process Image & Predict
if uploaded_file:
    with st.spinner("🔍 Analyzing image and identifying food item..."):
        image_pil = Image.open(uploaded_file).convert("RGB")
        prediction = predict_image(image_pil)
    st.success(f"🍴 Detected Dish: {prediction}")

    # Generate Recipe
    with st.spinner("🧠 Generating a step-by-step recipe using AI..."):
        recipe_text = get_recipe_steps(prediction)

    with st.expander("📋 View Full Recipe"):
        st.markdown(f"Here is your recipe for {prediction}:")
        st.text_area("Step-by-step Instructions", recipe_text, height=300)

    # Convert to Audio
    with st.spinner("🔊 Converting recipe to audio..."):
        audio_file = text_to_speech(recipe_text)

    st.subheader("🎧 Listen to the Recipe")
    st.audio(audio_file)

    st.download_button(
        label="⬇ Download Recipe Audio",
        data=open(audio_file, "rb"),
        file_name="recipe_audio.mp3",
        mime="audio/mpeg",
        help="Download the audio instructions"
    )

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Made with ❤ using Streamlit, TensorFlow & Cohere</p>", unsafe_allow_html=True)