import argparse, os, torch
import evaluate
from transformers import TrainingArguments, Trainer
from data import build_dataset, collate_fn
from models import build_model

def compute_metrics_builder():
    metric_auc = evaluate.load("roc_auc")
    metric_f1  = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        import numpy as np
        import torch
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        labels = np.asarray(labels)
        preds = (probs >= 0.5).astype(int)

        try:
            auc = metric_auc.compute(prediction_scores=probs, references=labels)["roc_auc"]
        except TypeError:
            auc = metric_auc.compute(predictions=probs, references=labels)["roc_auc"]
        f1  = metric_f1.compute(predictions=preds, references=labels, average="binary")["f1"]
        acc = (preds == labels).mean()
        return {"accuracy": float(acc), "auroc": float(auc), "f1": float(f1)}
    return compute_metrics

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="1aurent/PatchCamelyon")
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--model_name", default="google/vit-base-patch16-224")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out_dir", default="outputs/pcam_vit_base")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--num_workers", type=int, default=4)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337)

    train_ds, val_ds, test_ds, processor = build_dataset(
        dataset_id=args.dataset_id,
        cache_dir=args.cache_dir,
        img_size=args.img_size,
        model_name=args.model_name
    )

    model = build_model(model_name=args.model_name, num_labels=2)

    compute_metrics = compute_metrics_builder()

    args_hf = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        logging_steps=50,

        eval_strategy="epoch",
        logging_strategy="steps",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_auroc",
        greater_is_better=True,
        report_to=["tensorboard"],
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    trainer.train()

    val_metrics  = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    trainer.save_model(os.path.join(args.out_dir, "final_model"))

if __name__ == "__main__":
    main()
