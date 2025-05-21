import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('ai_art_detector.h5')

# Title
st.title("AI Art Fake Detector 🎨🤖")
st.write("Upload an image of artwork and find out if it's AI-generated or human-made!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "🧠 AI-Generated" if prediction < 0.5 else "🎨 Human-Made"

    st.markdown(f"### Prediction: {label}")
    st.markdown(f"Confidence: `{(1 - prediction if label == '🧠 AI-Generated' else prediction):.4f}`")

