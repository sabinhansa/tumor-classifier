import torch
from datasets import load_dataset, Image
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import random, os

MODEL_DIR = "outputs/pcam_vit_base/final_model"
MODEL_NAME = "google/vit-base-patch16-224"
DATASET_ID = "1aurent/PatchCamelyon"
CACHE_DIR = "data/cache"
IMG_SIZE = 224
DEVICE = "cuda"

print(f"Loading model from {MODEL_DIR}...")
model = ViTForImageClassification.from_pretrained(MODEL_DIR).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model.eval()

print("Loading test dataset...")
ds = load_dataset(DATASET_ID, cache_dir=CACHE_DIR)
if "test" in ds:
    test_ds = ds["test"]
else:
    test_ds = ds["train"].train_test_split(test_size=0.1, seed=1337)["test"]
test_ds = test_ds.cast_column("image", Image())

sample = test_ds[random.randint(0, len(test_ds) - 1)]
img = sample["image"]
true_label = int(sample["label"])

inputs = processor(images=img, size={"height": IMG_SIZE, "width": IMG_SIZE}, return_tensors="pt").to(DEVICE)

with torch.inference_mode():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_label = int(torch.argmax(probs))
    confidence = probs[pred_label].item()

id2label = model.config.id2label if hasattr(model.config, "id2label") else {0: "benign", 1: "malignant"}
pred_name = id2label[pred_label]
true_name = id2label[true_label]

print(f"True label: {true_name} ({true_label})")
print(f"Predicted: {pred_name} ({pred_label}) with confidence {confidence:.4f}")
print(f"Probabilities → benign={probs[0]:.4f}, malignant={probs[1]:.4f}")

plt.imshow(img)
plt.title(f"Pred: {pred_name} ({confidence:.2%}) | True: {true_name}")
plt.axis("off")
plt.show()
