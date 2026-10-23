"""Part 1.3 - fine-tune ImageNet VGG16 / AlexNet on MNIST (frozen features, new head)."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (AlexNet_Weights, VGG16_Weights, alexnet, vgg16)

from .data import MNISTTensor, RepeatChannels, load_mnist, make_loader
from .utils import (RESULTS, count_parameters, device_name, evaluate_classifier,
                    get_device, save_json, set_seed)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(name: str):
    """Pretrained ImageNet model with a new 10-class head and frozen conv features."""
    try:
        if name == "vgg16":
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            for p in model.features.parameters():
                p.requires_grad = False
            # torchvision VGG adaptively pools to 7x7, so only the last FC layer changes
            model.classifier[6] = nn.Linear(4096, 10)
        elif name == "alexnet":
            model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            for p in model.features.parameters():
                p.requires_grad = False
            model.classifier[6] = nn.Linear(4096, 10)
        else:
            raise ValueError(name)
        return model, True
    except Exception as exc:  # offline fallback: no pretrained weights
        print(f"[{name}] pretrained weights unavailable ({exc}); training from scratch")
        if name == "vgg16":
            model = vgg16(weights=None)
            for p in model.features.parameters():
                p.requires_grad = False
            model.classifier[6] = nn.Linear(4096, 10)
        else:
            model = alexnet(weights=None)
            for p in model.features.parameters():
                p.requires_grad = False
            model.classifier[6] = nn.Linear(4096, 10)
        return model, False


def main(model_name: str = "vgg16", epochs: int = 3, batch_size: int = 64, lr: float = 1e-3,
         image_size: int = 64, train_subset: int = 20000, test_subset: int | None = None,
         seed: int = 42) -> dict:
    set_seed(seed)
    device = get_device()
    tag = model_name
    print(f"[{tag}] device = {device_name(device)}, input={image_size}px")

    tr_x, tr_y, te_x, te_y = load_mnist()
    if train_subset:
        tr_x, tr_y = tr_x[:train_subset], tr_y[:train_subset]
    if test_subset:
        te_x, te_y = te_x[:test_subset], te_y[:test_subset]

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        RepeatChannels(3),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_loader = make_loader(MNISTTensor(tr_x, tr_y, tf), batch_size, shuffle=True)
    test_loader = make_loader(MNISTTensor(te_x, te_y, tf), batch_size, shuffle=False)

    model, pretrained = build_model(model_name)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable, lr=lr)

    history, train_time = [], 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        ev = evaluate_classifier(model, test_loader, criterion, device)
        history.append({
            "epoch": epoch, "train_loss": running / seen, "train_acc": correct / seen,
            "test_loss": ev["test_loss"], "test_acc": ev["test_acc"],
        })
        print(f"[{tag}] epoch {epoch}/{epochs} train_loss={running / seen:.4f} "
              f"train_acc={correct / seen:.4f} test_loss={ev['test_loss']:.4f} test_acc={ev['test_acc']:.4f}")

    final = evaluate_classifier(model, test_loader, criterion, device)
    result_out = {
        "model": "VGG16 (fine-tuned)" if model_name == "vgg16" else "AlexNet (fine-tuned)",
        "device": device_name(device),
        "pretrained_imagenet": pretrained,
        "frozen_features": True,
        "input_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": f"Adam(lr={lr}) on the new classifier only",
        "train_images": len(tr_x),
        "test_images": len(te_x),
        "params": count_parameters(model),
        "train_time_sec": train_time,
        "test_loss": final["test_loss"],
        "test_acc": final["test_acc"],
        "test_f1": final["test_f1"],
        "inference_ms_per_image": final["inference_ms_per_image"],
        "history": history,
    }
    np.savez_compressed(RESULTS / f"preds_{tag}.npz",
                        preds=final["preds"].astype(np.int16),
                        labels=final["labels"].astype(np.int16))
    save_json(result_out, RESULTS / f"{tag}.json")
    return result_out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", dest="model_name", choices=["vgg16", "alexnet"], default="vgg16")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--train-subset", type=int, default=20000)
    p.add_argument("--test-subset", type=int, default=None)
    a = p.parse_args()
    main(**vars(a))
