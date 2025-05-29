from utils.dino_features import extract_dino_feature_with_attention
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Step 1: Extract features and attention
features, attention = extract_dino_feature_with_attention(image)

# Step 2: Predict using your classifier
clf = joblib.load("model/svm_model.pkl")
probs = clf.predict_proba(features.reshape(1, -1))[0]
pred = clf.predict(features.reshape(1, -1))[0]

label = "🧠 Human-made" if pred == 0 else "🤖 AI-generated"
st.subheader(f"Prediction: {label}")
st.write(f"Confidence: {probs[pred]*100:.2f}%")

# Step 3: Prepare attention map
cls_attn = attention[0, 0, 0, 1:]  # [head=0, cls, rest]
num_patches = int(len(cls_attn) ** 0.5)
attn_map = cls_attn.reshape(num_patches, num_patches).detach().cpu().numpy()

# Normalize
attn_map -= attn_map.min()
attn_map /= attn_map.max()

# Resize to match original image
patch_size = 14  # vitb14
resized_attn = Image.fromarray(np.uint8(attn_map * 255)).resize(
    (image.width, image.height), resample=Image.BILINEAR
)
resized_attn_np = np.array(resized_attn)

# Step 4: Overlay attention
img_np = np.array(image.resize(resized_attn.size))
heatmap = cv2.applyColorMap(resized_attn_np, cv2.COLORMAP_JET)
overlayed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

# Step 5: Display overlay
st.subheader("🎯 Attention Overlay (CLS Token → Image Patches)")
st.image(overlayed, caption="Attention Map Overlay", use_column_width=True)