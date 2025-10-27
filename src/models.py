from transformers import ViTForImageClassification

def build_model(model_name: str = "google/vit-base-patch16-224", num_labels: int = 2):
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model
