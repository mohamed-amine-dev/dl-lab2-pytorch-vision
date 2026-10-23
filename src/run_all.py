"""Run the whole lab (all five models) and build the comparison artifacts.

Usage (from the repo root):
    python -m src.run_all                 # full run
    python -m src.run_all --quick         # tiny smoke test (~1 min)
    python -m src.run_all --only vit      # rerun one model
    python -m src.run_all --force         # ignore existing results
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import train_cnn, train_fasterrcnn, train_finetune, train_vit
from .plots import plot_comparison, plot_confusion, plot_curves, plot_vit_patches
from .utils import FIGURES, RESULTS, device_name, get_device, set_seed

# Budgets chosen for the machine that actually ran the lab (CPU, 12 threads).
FULL = {
    "cnn": dict(epochs=6, batch_size=256, lr=1e-3, weight_decay=1e-4, dropout=0.5),
    "fasterrcnn": dict(epochs=1, batch_size=4, lr=0.005, train_subset=3000,
                       test_subset=2000, curve_eval_images=300, min_size=96),
    "vgg16": dict(model_name="vgg16", epochs=2, batch_size=32, lr=3e-3,
                  image_size=64, train_subset=10000),
    "alexnet": dict(model_name="alexnet", epochs=3, batch_size=64, lr=3e-3,
                    image_size=64, train_subset=20000),
    "vit": dict(epochs=10, batch_size=256, lr=3e-4, weight_decay=0.01, patch_size=4,
                dim=64, depth=6, n_heads=4, mlp_ratio=4.0, dropout=0.1),
}

QUICK = {
    "cnn": dict(epochs=1, batch_size=128, train_subset=512, test_subset=256),
    "fasterrcnn": dict(epochs=1, batch_size=2, train_subset=24, test_subset=8,
                       curve_eval_images=4, min_size=64),
    "vgg16": dict(model_name="vgg16", epochs=1, batch_size=8, image_size=64,
                  train_subset=64, test_subset=32),
    "alexnet": dict(model_name="alexnet", epochs=1, batch_size=8, image_size=64,
                    train_subset=64, test_subset=32),
    "vit": dict(epochs=1, batch_size=64, dim=32, depth=2, n_heads=4,
                train_subset=128, test_subset=64),
}

FNS = {
    "cnn": train_cnn.main,
    "fasterrcnn": train_fasterrcnn.main,
    "vgg16": train_finetune.main,
    "alexnet": train_finetune.main,
    "vit": train_vit.main,
}

LABELS = {
    "cnn": "CNN", "fasterrcnn": "Faster R-CNN", "vgg16": "VGG16", "alexnet": "AlexNet",
    "vit": "ViT",
}


def build_report() -> pd.DataFrame:
    rows = []
    for key, label in LABELS.items():
        path = RESULTS / f"{key}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "key": key,
            "model": label,
            "test_acc": r["test_acc"],
            "test_f1": r["test_f1"],
            "test_loss": r.get("test_loss"),
            "train_time_sec": r["train_time_sec"],
            "params_total": r["params"]["params_total"],
            "params_trainable": r["params"]["params_trainable"],
            "inference_ms_per_image": r["inference_ms_per_image"],
            "train_images": r["train_images"],
            "test_images": r["test_images"],
            "epochs": r["epochs"],
            "device": r["device"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "comparison_all.csv", index=False, float_format="%.6f")
    df[df["key"].isin(["cnn", "fasterrcnn"])].to_csv(
        RESULTS / "comparison_part1.csv", index=False, float_format="%.6f")
    print(df.to_string(index=False))

    for key, label in LABELS.items():
        if key not in set(df["key"]):
            continue
        r = json.loads((RESULTS / f"{key}.json").read_text(encoding="utf-8"))
        plot_curves(r["history"], label, f"curves_{key}.png")
        p = RESULTS / f"preds_{key}.npz"
        if p.exists():
            z = np.load(p)
            plot_confusion(z["labels"], z["preds"], f"cm_{key}.png",
                           f"{label} - confusion matrix (test)")

    keys = set(df["key"])
    if {"cnn", "fasterrcnn"} <= keys:
        part1 = df[df["key"].isin(["cnn", "fasterrcnn"])]
        plot_comparison(part1, "comparison_part1.png", "Part 1 - CNN vs Faster R-CNN")
    if len(df) >= 2:
        plot_comparison(df, "comparison_all.png", "All models - MNIST")

    # ViT patch layout illustration
    from .data import load_mnist
    tr_x, _, _, _ = load_mnist()
    plot_vit_patches(tr_x[0], patch_size=4)

    print(f"[report] figures -> {FIGURES}, tables -> {RESULTS}")
    return df


def main(quick: bool = False, only: str | None = None, force: bool = False,
         skip_existing: bool = True) -> None:
    set_seed(42)
    device = get_device()
    print(f"[run_all] device = {device_name(device)}")
    cfg = QUICK if quick else FULL
    stages = [only] if only else list(FNS)

    for key in stages:
        out = RESULTS / f"{key}.json"
        if out.exists() and not force and skip_existing:
            print(f"[run_all] {key}: existing result found, skipping (use --force to redo)")
            continue
        print(f"\n===== {key} =====")
        FNS[key](**cfg[key])

    build_report()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--only", choices=list(FNS), default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-skip", dest="skip_existing", action="store_false")
    a = p.parse_args()
    main(quick=a.quick, only=a.only, force=a.force, skip_existing=a.skip_existing)
