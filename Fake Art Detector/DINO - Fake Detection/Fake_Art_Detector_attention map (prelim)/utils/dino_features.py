from transformers import AutoProcessor, AutoModel
import torch
import numpy as np
from PIL import Image

# Load processor and model
processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()

# Optional: create dummy classifier head (SVM, MLP, etc.) externally

def extract_dino_feature_with_attention(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # WARNING: DINOv2 on HF does NOT return attention maps
        outputs = model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        # Simulate prediction: you can connect this CLS token to any classifier
        logits = cls_token @ torch.randn(cls_token.shape[-1], 2)  # Example 2-class prediction
        probs = torch.softmax(logits, dim=-1)

    return cls_token.squeeze().numpy(), probs.squeeze().numpy()