"""MNIST loading (raw IDX parser with torchvision fallback) and dataset helpers."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------- IDX parsing
def _find(path: Path, names) -> Path:
    for n in names:
        p = path / n
        if p.is_file():
            return p
    for n in names:
        hits = [h for h in path.rglob(n) if h.is_file()]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"None of {names} found under {path}")


def read_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(root: Path | str = DATA_DIR / "mnist", cache: bool = True):
    """Return (train_images uint8 [N,28,28], train_labels, test_images, test_labels)."""
    root = Path(root)
    cache_file = root.parent / "mnist_arrays.npz"
    if cache and cache_file.exists():
        z = np.load(cache_file)
        return z["tr_x"], z["tr_y"], z["te_x"], z["te_y"]

    img_names = ["train-images-idx3-ubyte", "train-images.idx3-ubyte"]
    lbl_names = ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"]
    try:
        tr_x = read_idx_images(_find(root, img_names))
        tr_y = read_idx_labels(_find(root, lbl_names))
        te_x = read_idx_images(_find(root, ["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"]))
        te_y = read_idx_labels(_find(root, ["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"]))
        print(f"[data] raw IDX files loaded from {root}")
    except FileNotFoundError:
        print("[data] IDX files not found -> torchvision.datasets.MNIST(download=True)")
        tr = datasets.MNIST(str(root.parent), train=True, download=True)
        te = datasets.MNIST(str(root.parent), train=False, download=True)
        tr_x, tr_y = tr.data.numpy(), tr.targets.numpy()
        te_x, te_y = te.data.numpy(), te.targets.numpy()

    tr_x, tr_y, te_x, te_y = map(np.asarray, (tr_x, tr_y, te_x, te_y))
    if cache:
        np.savez_compressed(cache_file, tr_x=tr_x, tr_y=tr_y, te_x=te_x, te_y=te_y)
    print(f"[data] train={tr_x.shape}, test={te_x.shape}")
    return tr_x, tr_y, te_x, te_y


# ---------------------------------------------------------------- datasets
class MNISTTensor(Dataset):
    """MNIST kept in memory as float tensors in [0,1]; a transform is applied per item."""

    def __init__(self, images, labels, transform=None):
        self.images = torch.from_numpy(np.asarray(images)).float().unsqueeze(1) / 255.0
        self.labels = torch.from_numpy(np.asarray(labels)).long()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


class RepeatChannels:
    """1-channel [0,1] tensor -> 3-channel tensor (for ImageNet models)."""

    def __init__(self, n: int = 3):
        self.n = n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(self.n, 1, 1)


class MNISTDetection(Dataset):
    """MNIST as an object-detection dataset: tight box around the ink, label = digit+1."""

    def __init__(self, images, labels, ink_threshold: float = 0.2):
        self.images = torch.from_numpy(np.asarray(images)).float().unsqueeze(1) / 255.0
        self.labels = torch.from_numpy(np.asarray(labels)).long()
        self.thr = ink_threshold

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        mask = img[0] > self.thr
        rows = torch.nonzero(mask.any(1)).flatten()
        cols = torch.nonzero(mask.any(0)).flatten()
        if rows.numel() == 0:  # blank image: fall back to the whole frame
            box = torch.tensor([[0.0, 0.0, 28.0, 28.0]])
        else:
            box = torch.tensor(
                [[cols.min().item(), rows.min().item(),
                  (cols.max() + 1).item(), (rows.max() + 1).item()]], dtype=torch.float32
            )
        target = {"boxes": box, "labels": torch.tensor([self.labels[i].item() + 1], dtype=torch.int64)}
        return img.repeat(3, 1, 1), target


def detection_collate(batch):
    return tuple(zip(*batch))


def make_loader(dataset, batch_size: int, shuffle: bool = False, collate_fn=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows-safe, also fine on Kaggle
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
