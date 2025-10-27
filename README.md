# Tumor Patch Classifier (ViT – PatchCamelyon Dataset)

## Overview
This project implements a binary image classifier that distinguishes between **benign** and **malignant** histopathology patches from the *PatchCamelyon* dataset.  
The model uses a **Vision Transformer (ViT)** architecture fine-tuned on the dataset using the **Hugging Face Transformers** framework.  
It includes complete data preparation, training, evaluation, and inference pipelines.

---

## Project Structure

```
tumor-classifier/
│
├── src/
│ ├── train.py # Model training and evaluation
│ ├── infer.py # Inference on single images
│ ├── models.py # Model construction and configuration
│ ├── data.py # Dataset loading and preprocessing
│ ├── eval.py # Evaluation metrics and utilities
│
├── scripts/
│ └── preview_dataset.py # Script used to visualize dataset samples
│
├── data/ # (ignored) dataset cache and raw files
├── outputs/ # (ignored) model checkpoints and training logs
│
├── preview_class0.png
├── preview_class1.png
├── preview_train_grid.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset
The model uses the [**PatchCamelyon (PCam)**](https://huggingface.co/datasets/1aurent/PatchCamelyon) dataset, derived from the **Camelyon16** challenge.  
Each 96×96 RGB patch is labeled as:

- **0 – benign (no tumor tissue)**  
- **1 – malignant (tumor tissue)**

---

## Model
- **Architecture:** `google/vit-base-patch16-224`  
- **Frameworks:** PyTorch, Hugging Face Transformers, datasets, and evaluate  
- **Objective:** Fine-tuned for binary classification using cross-entropy loss  
- **Metrics:** Accuracy, F1-score, and AUROC  

---

## Training
The model is trained using `train.py`, which:

1. Loads and preprocesses the dataset with automatic stratified splitting  
2. Applies on-the-fly image transformation and batching  
3. Fine-tunes the ViT backbone for several epochs  
4. Evaluates performance each epoch and saves the best checkpoint  

## Example Command
```python src/train.py --dataset_id "1aurent/PatchCamelyon" --fp16 --epochs 5 --batch_size 32```

---

## Example Validation Results
- **Accuracy:** 0.9866  
- **AUROC:** 0.9983  
- **F1-score:** 0.9866  

---

## Evaluation and Inference
After training, you can evaluate or run inference on new images:
```python src/infer.py --image path/to/image.png --model_dir outputs/pcam_vit_base/final_model```

---

## Example Dataset Samples
Sample visualizations from the training dataset can be found at preview_class0.png, preview_class1.png, preview_train_grid.png

---

## Reproducibility
- **Install dependencies:** ```pip install -r requirements.txt```
- **Train from scratch:**```python src/train.py```
- **Evaluate or visualize results with:**```python src/eval.py```,```python scripts/preview_dataset.py```
