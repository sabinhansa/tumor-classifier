import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

def pick_threshold_on_val(trainer, val_ds, step=0.05):
    out = trainer.predict(val_ds)
    probs = torch.softmax(torch.tensor(out.predictions), dim=-1)[:, 1].numpy()
    labels = out.label_ids.astype(int)

    ths = np.arange(step, 1.0, step)
    scores = [f1_score(labels, (probs >= t).astype(int)) for t in ths]
    best_idx = int(np.argmax(scores))
    return float(ths[best_idx]), float(scores[best_idx])

def evaluate_with_threshold(trainer, ds, threshold: float, name: str = "eval"):
    out = trainer.predict(ds)
    probs = torch.softmax(torch.tensor(out.predictions), dim=-1)[:, 1].numpy()
    labels = out.label_ids.astype(int)
    preds = (probs >= threshold).astype(int)
    auroc = roc_auc_score(labels, probs)
    f1    = f1_score(labels, preds)
    cm    = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, digits=4)
    print(f"\n[{name}] threshold={threshold:.3f} AUROC={auroc:.4f} F1={f1:.4f}")
    print(f"[{name}] confusion matrix:\n{cm}")
    print(f"[{name}] report:\n{report}")
    return {"auroc": float(auroc), "f1": float(f1), "threshold": float(threshold)}
