import random
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset, Image

CACHE_DIR = "data/cache"               # top-level cache dir
DATASET_ID = "1aurent/PatchCamelyon"   # your cached dataset

def grid(pairs, out_path, title):
    cols = 8
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, lab) in zip(axes.ravel(), pairs):
        ax.imshow(img)
        ax.set_title(f"label={lab}", fontsize=8)
        ax.axis("off")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved", out_path)

if __name__ == "__main__":
    Path("outputs").mkdir(parents=True, exist_ok=True)
    random.seed(1337)

    # Load from HF cache and ensure the 'image' column is an Image() feature
    ds = load_dataset(DATASET_ID, cache_dir=CACHE_DIR)
    ds = ds.cast_column("image", Image())  # ensures decoding on access
    ds.set_format(type="python")           # return native Python objects (PIL for Image)

    # Random train grid
    idxs = random.sample(range(len(ds["train"])), k=32)
    samples = [(ds["train"][i]["image"], int(ds["train"][i]["label"])) for i in idxs]
    grid(samples, "outputs/preview_train_grid.png", "PCam train preview (HF)")

    # First 16 per class
    zeros, ones = [], []
    for ex in ds["train"]:
        (zeros if ex["label"] == 0 else ones).append((ex["image"], int(ex["label"])))
        if len(zeros) >= 16 and len(ones) >= 16:
            break
    grid(zeros[:16], "outputs/preview_class0.png", "label=0 (benign) examples")
    grid(ones[:16], "outputs/preview_class1.png", "label=1 (malignant) examples")
