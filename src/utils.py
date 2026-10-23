"""Shared helpers: seeding, device, metrics, timers, JSON output."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"[saved] {path}")


@torch.no_grad()
def evaluate_classifier(model, loader, criterion, device, max_batches=None):
    """Return loss, accuracy, macro-F1, predictions, labels and seconds/image."""
    model.eval()
    total_loss, n = 0.0, 0
    preds, labels = [], []
    sync(device)
    start = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    sync(device)
    elapsed = time.perf_counter() - start
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return {
        "test_loss": total_loss / n,
        "test_acc": float(accuracy_score(labels, preds)),
        "test_f1": float(f1_score(labels, preds, average="macro")),
        "preds": preds,
        "labels": labels,
        "inference_ms_per_image": 1000.0 * elapsed / n,
        "eval_seconds": elapsed,
        "n_eval": int(n),
    }


def finalize_result(result: dict, preds_path: Path) -> dict:
    """Store predictions separately so the JSON stays small."""
    preds = result.pop("preds")
    labels = result.pop("labels")
    np.savez_compressed(preds_path, preds=preds.astype(np.int16), labels=labels.astype(np.int16))
    return result
