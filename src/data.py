import torch
from datasets import load_dataset, Image, ClassLabel
from transformers import AutoImageProcessor
from torchvision import transforms as T
from PIL import Image as PILImage
from typing import Tuple, Optional

class HFImageTransform:
    def __init__(self, model_name: str, img_size: int, use_fast: bool = True, train: bool = False):
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=use_fast)
        self.size = img_size
        self.train = train
        self.aug = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ])

    def _maybe_aug(self, img: PILImage.Image):
        return self.aug(img) if self.train else img

    def __call__(self, example):
        imgs = example["image"]
        is_batch = isinstance(imgs, list)

        if is_batch:
            imgs = [self._maybe_aug(im) for im in imgs]
        else:
            imgs = self._maybe_aug(imgs)

        inputs = self.processor(
            images=imgs,
            do_resize=not self.train,
            size={"height": self.size, "width": self.size},
            return_tensors="pt",
        )

        if is_batch:
            example["pixel_values"] = [pv for pv in inputs["pixel_values"]]
            labs = example["label"]
            example["labels"] = [int(x) for x in labs]
        else:
            example["pixel_values"] = inputs["pixel_values"][0]
            example["labels"] = int(example["label"])
        return example


def build_dataset(
    dataset_id: str,
    cache_dir: str,
    img_size: int,
    model_name: str,
    seed: int = 1337,
    val_frac: float = 0.1
):
    ds = load_dataset(dataset_id, cache_dir=cache_dir)

    if "validation" in ds:
        ds_train, ds_val = ds["train"], ds["validation"]
    elif "val" in ds:
        ds_train, ds_val = ds["train"], ds["val"]
    else:
        ds_train, ds_val = ds["train"], None

    if "label" in ds_train.features and not isinstance(ds_train.features["label"], ClassLabel):
        cl = ClassLabel(num_classes=2, names=["benign", "malignant"])
        for k in ds.keys():
            ds[k] = ds[k].cast_column("label", cl)
        ds_train = ds["train"]
        if ds_val is not None:
            ds_val = ds["validation"] if "validation" in ds else ds["val"]

    if ds_val is None:
        split = ds_train.train_test_split(test_size=val_frac, seed=seed, stratify_by_column="label")
        ds_train, ds_val = split["train"], split["test"]

    ds_test = ds["test"] if "test" in ds else ds_val

    ds_train = ds_train.cast_column("image", Image())
    ds_val   = ds_val.cast_column("image", Image())
    ds_test  = ds_test.cast_column("image", Image())

    tf_train = HFImageTransform(model_name, img_size, use_fast=True, train=True)
    tf_eval  = HFImageTransform(model_name, img_size, use_fast=True, train=False)

    ds_train = ds_train.with_transform(tf_train)
    ds_val   = ds_val.with_transform(tf_eval)
    ds_test  = ds_test.with_transform(tf_eval)

    return ds_train, ds_val, ds_test, tf_eval.processor


def collate_fn(batch):
    x = torch.stack([b["pixel_values"] for b in batch])
    y = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"pixel_values": x, "labels": y}
