"""Part 1.2 - Faster R-CNN adapted to MNIST (digit detection instead of classification)."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .data import MNISTDetection, detection_collate, load_mnist, make_loader
from .utils import (RESULTS, count_parameters, device_name, get_device, save_json,
                    set_seed, sync)


def build_model(num_classes: int = 11, min_size: int = 128, max_size: int = 192):
    """COCO-pretrained MobileNetV3 Faster R-CNN with a fresh 11-class head."""
    pretrained = True
    try:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
            weights_backbone=None,
        )
    except Exception as exc:  # offline fallback
        print(f"[fasterrcnn] pretrained weights unavailable ({exc}); training from scratch")
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
        pretrained = False
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # smaller images than the default 320px -> much faster on a small dataset
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    return model, pretrained


@torch.no_grad()
def predict_detection(model, loader, device, max_images=None, timed=True):
    """Predict the digit of the highest-scoring box; background (10) if nothing detected."""
    model.eval()
    preds, labels = [], []
    sync(device)
    start = time.perf_counter()
    n = 0
    for images, targets in loader:
        if max_images is not None and n >= max_images:
            break
        images = [im.to(device) for im in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            if len(out["scores"]) == 0:
                pred = 10  # background -> counts as a wrong classification
            else:
                pred = int(out["labels"][out["scores"].argmax()].item()) - 1
            preds.append(pred)
            labels.append(int(tgt["labels"][0].item()) - 1)
        n += len(images)
    sync(device)
    elapsed = time.perf_counter() - start
    preds, labels = np.array(preds), np.array(labels)
    acc = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, labels=list(range(10)), average="macro", zero_division=0))
    return {
        "preds": preds,
        "labels": labels,
        "test_acc": acc,
        "test_f1": f1,
        "n_background": int((preds == 10).sum()),
        "inference_ms_per_image": (1000.0 * elapsed / len(labels)) if timed else None,
    }


def main(epochs: int = 2, batch_size: int = 8, lr: float = 0.005, momentum: float = 0.9,
         weight_decay: float = 5e-4, train_subset: int = 5000, test_subset: int = 2000,
         curve_eval_images: int = 500, min_size: int = 128, seed: int = 42,
         tag: str = "fasterrcnn") -> dict:
    set_seed(seed)
    device = get_device()
    print(f"[{tag}] device = {device_name(device)}")

    tr_x, tr_y, te_x, te_y = load_mnist()
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(tr_x), size=min(train_subset, len(tr_x)), replace=False)
    te_idx = np.arange(min(test_subset, len(te_x)))  # deterministic: first N test images
    tr_x, tr_y = tr_x[tr_idx], tr_y[tr_idx]
    te_x, te_y = te_x[te_idx], te_y[te_idx]

    train_ds = MNISTDetection(tr_x, tr_y)
    test_ds = MNISTDetection(te_x, te_y)
    train_loader = make_loader(train_ds, batch_size, shuffle=True, collate_fn=detection_collate)
    test_loader = make_loader(test_ds, batch_size, shuffle=False, collate_fn=detection_collate)
    curve_loader = make_loader(test_ds, batch_size, shuffle=False, collate_fn=detection_collate)

    model, pretrained = build_model(min_size=min_size)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    history, train_time = [], 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for i, (images, targets) in enumerate(train_loader):
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            seen += len(images)
            if (i + 1) % 50 == 0:
                print(f"[{tag}] epoch {epoch} iter {i + 1}/{len(train_loader)} loss={running / (i + 1):.4f}")
        sync(device)
        train_time += time.perf_counter() - t0
        scheduler.step()

        ev = predict_detection(model, curve_loader, device, max_images=curve_eval_images, timed=False)
        history.append({
            "epoch": epoch,
            "train_loss": running / len(train_loader),
            "train_acc": np.nan,            # detection loss is a sum of 4 terms -> not comparable to CE
            "test_loss": np.nan,            # test CE is not defined for a detector (see README)
            "test_acc": ev["test_acc"],
        })
        print(f"[{tag}] epoch {epoch}/{epochs} detection_loss={running / len(train_loader):.4f} "
              f"acc@{curve_eval_images}={ev['test_acc']:.4f}")

    final = predict_detection(model, test_loader, device)
    result = {
        "model": "Faster R-CNN (MobileNetV3)",
        "device": device_name(device),
        "pretrained_backbone": pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": f"SGD(lr={lr}, momentum={momentum}, weight_decay={weight_decay})",
        "input_resize": f"min_size={min_size}, max_size=192",
        "train_images": len(train_ds),
        "test_images": len(test_ds),
        "params": count_parameters(model),
        "train_time_sec": train_time,
        "test_loss": None,
        "test_acc": final["test_acc"],
        "test_f1": final["test_f1"],
        "n_background_predictions": final["n_background"],
        "inference_ms_per_image": final["inference_ms_per_image"],
        "history": history,
        "notes": ("Detection model adapted to classification: box around the ink, label = digit+1, "
                  "highest-scoring box used as prediction; train/test loss curves are detection losses "
                  "(sum of RPN + ROI losses) and no test cross-entropy exists."),
    }
    np.savez_compressed(RESULTS / f"preds_{tag}.npz",
                         preds=final["preds"].astype(np.int16),
                         labels=final["labels"].astype(np.int16))
    save_json(result, RESULTS / f"{tag}.json")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--train-subset", type=int, default=5000)
    p.add_argument("--test-subset", type=int, default=2000)
    p.add_argument("--curve-eval-images", type=int, default=500)
    p.add_argument("--min-size", type=int, default=128)
    a = p.parse_args()
    main(**vars(a))
