import streamlit as st
import joblib
from PIL import Image
from utils.dino_features import extract_dino_feature
import numpy as np
import matplotlib.pyplot as plt

# Page title
st.set_page_config(page_title="AI Art Fake Detection", layout="centered")
st.title("🎨 AI Art Fake Detection App")
st.write("Upload an artwork to predict whether it's **human-made** or **AI-generated** using DINOv2 features + SVM classifier.")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Artwork", use_column_width=True)

    st.info("🔍 Extracting features using DINOv2...")
    features = extract_dino_feature(image).reshape(1, -1)

    # Load trained model
    clf = joblib.load("model/svm_model.pkl")

    # Prediction
    prob = clf.predict_proba(features)[0]
    pred = clf.predict(features)[0]

    # Output result
    label = "🧠 Human-made" if pred == 0 else "🤖 AI-generated"
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {prob[pred]*100:.2f}%")

    # Confidence bar chart
    st.write("### 🔎 Prediction Confidence")
    classes = ["Human-made", "AI-generated"]
    colors = ["#4CAF50", "#FF5722"]

    fig, ax = plt.subplots()
    bars = ax.bar(classes, prob * 100, color=colors)
    ax.set_ylim([0, 100])
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Prediction Confidence")

    # Annotate bars
    for bar, p in zip(bars, prob):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{p*100:.1f}%", ha='center')

    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built using DINOv2, Streamlit, and scikit-learn.")