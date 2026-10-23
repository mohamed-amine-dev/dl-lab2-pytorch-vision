"""Plot helpers: training curves, comparison bars, confusion matrices, ViT patches."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .utils import FIGURES


def plot_curves(history, title: str, filename: str) -> Path:
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], "o-", label="train")
    if "test_loss" in df and df["test_loss"].notna().any():
        axes[0].plot(df["epoch"], df["test_loss"], "s-", label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{title} - loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"] * 100, "o-", label="train")
    if "test_acc" in df and df["test_acc"].notna().any():
        axes[1].plot(df["epoch"], df["test_acc"] * 100, "s-", label="test")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title(f"{title} - accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_comparison(df: pd.DataFrame, filename: str, title: str = "Model comparison") -> Path:
    metrics = [
        ("test_acc", "Test accuracy (%)", 100.0),
        ("test_f1", "Macro F1 (%)", 100.0),
        ("train_time_sec", "Training time (s)", 1.0),
        ("params_total", "Parameters (millions)", 1e-6),
        ("inference_ms_per_image", "Inference (ms/image)", 1.0),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    for ax, (col, label, scale) in zip(axes, metrics):
        vals = df[col].fillna(0) * scale
        bars = ax.bar(df["model"], vals, color=sns.color_palette("deep", len(df)))
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            txt = f"{v:.3f}" if scale == 1e-6 else f"{v:.2f}"
            ax.text(b.get_x() + b.get_width() / 2, v, txt, ha="center", va="bottom", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(labels, preds, filename: str, title: str, classes=None) -> Path:
    classes = classes or list(range(10))
    cm = confusion_matrix(labels, preds, labels=classes)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_vit_patches(image: np.ndarray, patch_size: int, filename: str = "vit_patch_layout.png") -> Path:
    """Show one MNIST image next to its patch grid (ViT token layout)."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("input 28x28")
    axes[0].axis("off")
    axes[1].imshow(image, cmap="gray")
    n = 28 // patch_size
    for i in range(n + 1):
        axes[1].axhline(i * patch_size - 0.5, color="tab:red", lw=1)
        axes[1].axvline(i * patch_size - 0.5, color="tab:red", lw=1)
    axes[1].set_title(f"{n * n} patches of {patch_size}x{patch_size} + [CLS]")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
