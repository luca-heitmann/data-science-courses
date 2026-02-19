from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms.functional import to_pil_image

ORIGINAL_IMAGE = "/Users/luca/Projects/ms-data-science/deep-learning/hw11/output.png"

# Load model
weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1.DEFAULT
preprocess = weights.transforms()
model = efficientnet_v2_s(weights=weights)
model.eval()

# Predict image
with Image.open(ORIGINAL_IMAGE) as img:
  # Apply inference preprocessing transforms
  batch = preprocess(img).unsqueeze(0)
  
  # Use the model and print the predicted category
  prediction = model(batch).squeeze(0).softmax(0)
  class_id = prediction.argmax().item()
  score = prediction[class_id].item()
  category_name = weights.meta["categories"][class_id]

  print(category_name)
