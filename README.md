# CV_Final_Project

This repository contains the final group project for the Advanced Computer Vision course. Our project explores two core computer vision challenges applied to an art dataset:
	•	Style Transfer
	•	Fake Image Detection

We implement and compare two different models for each task, identifying a “champion” (best-performing) and a “challenger” (alternative) based on performance and deployment readiness. The project concludes with real-time model deployment using Streamlit.


🔍 Project Overview

🎨 1. Style Transfer

We use deep learning to transfer artistic styles to input images.

Models:
	•	Champion: Custom CNN built from scratch with content and style loss optimization
	•	Challenger: Fast Neural Style Transfer

Evaluation Criteria: Visual output, inference time, content-style tradeoff


🖼️ 2. Fake Image Detection

We classify whether an artwork is AI-generated or human-made using two architectures and two dataset configurations (pure vs. hybrid).

Models:
	•	Champion: EfficientNet
	•	Challenger: DINOv2 with a regression head for classification

Evaluation Metrics: Accuracy, precision, recall, F1-score, confusion matrix

🧠 Models

We implemented and compared two models for each task, designating a Champion (best performing) and a Challenger (alternative approach).

🎨 Style Transfer
	•	Champion: VGG-based Neural Style Transfer using a pretrained model and Cubism-style reference image
	•	Challenger: Custom CNN architecture trained from scratch

🖼️ Fake Art Detection
	•	Champion: Pre-trained DINOv2 with a custom classification head (linear regression for logits)
	•	Challenger: EfficientNet (adapted from public Kaggle code and Medium resources)


⚙️ Methodology
	•	Preprocessing: Image normalization, resizing, label encoding
	•	Training: Implemented with PyTorch, early stopping and LR scheduling used
	•	Loss Functions:
	•	Style Transfer: Content loss + Style loss
	•	Fake Detection: Binary Cross-Entropy Loss
	•	Evaluation: Visual and quantitative metrics, ROC curves for fake detection
	•	Deployment: Streamlit app

