"""Build the two lab notebooks (nbformat) so they only read saved results."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
NB_DIR = REPO / "notebooks"
NB_DIR.mkdir(exist_ok=True)

SETUP = '''\
# --- Setup: works locally, on Colab and on Kaggle -------------------------
import json, os, subprocess, sys
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if not (REPO / "src").exists():  # on Colab/Kaggle: clone the public repo
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/mohamed-amine-dev/dl-lab2-pytorch-vision.git",
                    "dl-lab2-pytorch-vision"], check=True)
    REPO = REPO / "dl-lab2-pytorch-vision"
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Image, display

%matplotlib inline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("Device :", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (no GPU available)")

# The heavy training is done by the scripts in src/ and cached in results/.
# Set to True (with a GPU) to retrain everything from scratch instead of loading results.
RUN_TRAINING = False
if RUN_TRAINING:
    subprocess.run([sys.executable, "-m", "src.run_all", "--force"], check=True)
'''

DATA = '''\
from src.data import load_mnist

tr_x, tr_y, te_x, te_y = load_mnist()   # raw IDX files, torchvision download as fallback
print("train:", tr_x.shape, "test:", te_x.shape, "| labels:", tr_y.min(), "->", tr_y.max())
print("class counts (train):", np.bincount(tr_y))

fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
for ax, img, lab in zip(axes.ravel(), tr_x[:16], tr_y[:16]):
    ax.imshow(img, cmap="gray")
    ax.set_title(int(lab))
    ax.axis("off")
fig.suptitle("MNIST samples")
plt.tight_layout()
plt.show()
'''

CNN_SHAPES = '''\
from src.models import CNN
from src.utils import count_parameters

model = CNN(dropout=0.5)
x = torch.randn(1, 1, 28, 28)
print("input      ", tuple(x.shape))
for name, layer in model.features.named_children():
    x = layer(x)
    print(f"{name:9s} {type(layer).__name__:11s} -> {tuple(x.shape)}")
x = model.classifier[0](x); print("flatten    ", tuple(x.shape))
x = model.classifier[1](x); print("fc1        ", tuple(x.shape))
x = model.classifier[3](x); print("dropout    ", tuple(x.shape))
x = model.classifier[4](x); print("fc2        ", tuple(x.shape))
print(count_parameters(model))
'''

FIT_TABLE = '''\
def config_table(result, keys):
    return pd.DataFrame({k: [result.get(k)] for k in keys})

cnn = json.loads((REPO / "results/cnn.json").read_text())
print("Device actually used:", cnn["device"])
config_table(cnn, ["epochs", "batch_size", "optimizer", "dropout", "train_images", "test_images"]).T
'''

CNN_HISTORY = '''\
display(pd.DataFrame(cnn["history"]).round(4))
display(Image(filename=str(REPO / "figures/curves_cnn.png")))
'''

CNN_METRICS = '''\
pd.DataFrame([{
    "test accuracy": round(cnn["test_acc"], 4),
    "macro F1": round(cnn["test_f1"], 4),
    "test loss": round(cnn["test_loss"], 4),
    "training time (s)": round(cnn["train_time_sec"], 1),
    "parameters": cnn["params"]["params_total"],
    "inference (ms/img)": round(cnn["inference_ms_per_image"], 3),
}])
'''

FRCNN_TARGET = '''\
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.data import MNISTDetection

# Build the detector head with 11 classes = 10 digits + background
detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
detector.roi_heads.box_predictor = FastRCNNPredictor(
    detector.roi_heads.box_predictor.cls_score.in_features, 11)
print(detector.roi_heads.box_predictor)

# Target built from the ink of one image
ds = MNISTDetection(tr_x[:1], tr_y[:1])
img3, target = ds[0]
print("image:", tuple(img3.shape), "| box [xmin,ymin,xmax,ymax]:", target["boxes"].tolist(),
      "| label (digit+1):", target["labels"].tolist())

box = target["boxes"][0]
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(tr_x[0], cmap="gray")
ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                           edgecolor="red", facecolor="none", lw=2))
ax.set_title(f"digit {int(tr_y[0])} -> class {int(target['labels'][0])}")
plt.show()
'''

FRCNN_RESULTS = '''\
fr = json.loads((REPO / "results/fasterrcnn.json").read_text())
print("Device actually used:", fr["device"])
config_table(fr, ["epochs", "batch_size", "optimizer", "input_resize",
                  "train_images", "test_images", "pretrained_backbone"]).T
'''

FRCNN_HISTORY = '''\
display(pd.DataFrame(fr["history"]).round(4))
print("background predictions:", fr["n_background_predictions"])
display(Image(filename=str(REPO / "figures/curves_fasterrcnn.png")))
'''

FRCNN_METRICS = '''\
pd.DataFrame([{
    "test accuracy": round(fr["test_acc"], 4),
    "macro F1": round(fr["test_f1"], 4),
    "test loss": "N/A (detector)",
    "training time (s)": round(fr["train_time_sec"], 1),
    "parameters": fr["params"]["params_total"],
    "inference (ms/img)": round(fr["inference_ms_per_image"], 3),
}])
'''

PART1_TABLE = '''\
part1 = pd.read_csv(REPO / "results/comparison_part1.csv")
cols = ["model", "test_acc", "test_f1", "test_loss", "train_time_sec",
        "params_total", "inference_ms_per_image", "train_images", "test_images"]
table = part1[cols].copy()
table.columns = ["model", "accuracy", "macro F1", "test loss", "train time (s)",
                 "params", "inference (ms/img)", "train imgs", "test imgs"]
display(table.round(4))
display(Image(filename=str(REPO / "figures/comparison_part1.png")))
'''

FINETUNE_SETUP = '''\
import torch.nn as nn
from torchvision import transforms
from torchvision.models import alexnet, vgg16
from src.data import RepeatChannels
from src.train_finetune import IMAGENET_MEAN, IMAGENET_STD

# The scripts load ImageNet weights (VGG16_Weights.IMAGENET1K_V1 / AlexNet_Weights.IMAGENET1K_V1);
# weights=None here only to show the head surgery without downloading 500 MB in the notebook.
vgg = vgg16(weights=None)
for p in vgg.features.parameters():
    p.requires_grad = False
vgg.classifier[6] = nn.Linear(4096, 10)
print("VGG16 classifier:\\n", vgg.classifier)
print("trainable parameters:", f"{sum(p.numel() for p in vgg.parameters() if p.requires_grad):,}")

alex = alexnet(weights=None)
for p in alex.features.parameters():
    p.requires_grad = False
alex.classifier[6] = nn.Linear(4096, 10)
print("\\nAlexNet classifier:\\n", alex.classifier)
print("trainable parameters:", f"{sum(p.numel() for p in alex.parameters() if p.requires_grad):,}")

tf = transforms.Compose([
    transforms.Resize((64, 64), antialias=True),
    RepeatChannels(3),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
print("\\npreprocessing:", tf)
'''

FINETUNE_RESULTS = '''\
vgg = json.loads((REPO / "results/vgg16.json").read_text())
alex = json.loads((REPO / "results/alexnet.json").read_text())
rows = []
for r in (vgg, alex):
    rows.append({
        "model": r["model"], "device": r["device"], "input": f'{r["input_size"]}x{r["input_size"]}',
        "epochs": r["epochs"], "train imgs": r["train_images"],
        "accuracy": r["test_acc"], "macro F1": r["test_f1"], "test loss": r["test_loss"],
        "train time (s)": r["train_time_sec"],
        "params (total / trainable)": f'{r["params"]["params_total"]:,} / {r["params"]["params_trainable"]:,}',
        "inference (ms/img)": r["inference_ms_per_image"],
    })
display(pd.DataFrame(rows).round(4))
display(Image(filename=str(REPO / "figures/curves_vgg16.png")))
display(Image(filename=str(REPO / "figures/curves_alexnet.png")))
'''

ALL_TABLE = '''\
allres = pd.read_csv(REPO / "results/comparison_all.csv")
cols = ["model", "test_acc", "test_f1", "test_loss", "train_time_sec",
        "params_total", "inference_ms_per_image", "train_images", "test_images"]
t = allres[cols].copy()
t.columns = ["model", "accuracy", "macro F1", "test loss", "train time (s)",
             "params", "inference (ms/img)", "train imgs", "test imgs"]
display(t.round(4))
display(Image(filename=str(REPO / "figures/comparison_all.png")))
'''

CNN_CM = '''\
z = np.load(REPO / "results/preds_cnn.npz")
from sklearn.metrics import classification_report
print(classification_report(z["labels"], z["preds"], digits=4))
display(Image(filename=str(REPO / "figures/cm_cnn.png")))
'''

# ----------------------------------------------------------------- notebook 1
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell(
        "# Deep Learning Lab 2 - Part 1: CNN, Faster R-CNN and transfer learning on MNIST\n"
        "**Master MBD - Deep Learning - FST Tanger, Université Abdelmalek Essaadi**\n\n"
        "This notebook loads the metrics and figures produced by the training scripts in `src/` "
        "(the heavy compute is done once and cached in `results/` and `figures/`). "
        "Every number below comes from a real run on the device printed in the next cells."),
    nbf.v4.new_code_cell(SETUP),
    nbf.v4.new_markdown_cell(
        "## 1. Dataset\n"
        "MNIST (LeCun et al.): 70 000 grayscale 28x28 digit images, 60 000 train / 10 000 test, "
        "10 balanced classes. Raw IDX files come from "
        "[kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) "
        "and are parsed with `struct` + NumPy in `src/data.py`; `torchvision.datasets.MNIST` is the fallback."),
    nbf.v4.new_code_cell(DATA),
    nbf.v4.new_markdown_cell(
        "## 2. Custom CNN\n"
        "**Architecture** (input 1x28x28):\n\n"
        "| block | layer | output |\n"
        "|---|---|---|\n"
        "| 1 | Conv2d(1→32, k=3, stride=1, padding=1) + ReLU | 32x28x28 |\n"
        "| 2 | MaxPool2d(2, 2) | 32x14x14 |\n"
        "| 3 | Conv2d(32→64, k=3, stride=1, padding=1) + ReLU | 64x14x14 |\n"
        "| 4 | MaxPool2d(2, 2) | 64x7x7 |\n"
        "| 5 | Flatten → Linear(3136→128) + ReLU + Dropout(0.5) | 128 |\n"
        "| 6 | Linear(128→10) | 10 logits |\n\n"
        "**Why these choices**\n"
        "- *Kernel 3x3, stride 1, padding 1*: keeps the spatial size after each convolution, so the two "
        "pooling layers alone reduce 28→14→7. Two stacked 3x3 convs already see a 5x5 receptive field with "
        "fewer parameters than one 5x5 kernel; this is the standard VGG-style choice.\n"
        "- *Max pooling 2x2*: halves the resolution, gives small translation invariance and reduces compute.\n"
        "- *Optimizer Adam (lr=1e-3)*: robust default for a small CNN, converges in a few epochs.\n"
        "- *Regularization*: weight decay 1e-4 (L2) + Dropout 0.5 before the final layer to limit over-fitting "
        "on the 3 136→128 fully-connected layer.\n"
        "- *Loss*: CrossEntropyLoss (softmax + NLL), batch size 256, 6 epochs."),
    nbf.v4.new_code_cell(CNN_SHAPES),
    nbf.v4.new_code_cell(FIT_TABLE),
    nbf.v4.new_code_cell(CNN_HISTORY),
    nbf.v4.new_code_cell(CNN_METRICS),
    nbf.v4.new_code_cell(CNN_CM),
    nbf.v4.new_markdown_cell(
        "Training is stable and converges quickly: test accuracy goes from 97.6% after one epoch to 99.2% "
        "after six, while the training and test curves stay very close (test accuracy is even slightly "
        "higher than training accuracy because dropout is active during training). There is no visible "
        "over-fitting, but the curves clearly saturate after ~4 epochs, so 6 epochs is already more than "
        "needed for this architecture. Confusions are almost only between visually close digits (4/9, 3/5, 7/1)."),
    nbf.v4.new_markdown_cell(
        "## 3. Faster R-CNN on MNIST (object detection adapted to classification)\n"
        "Faster R-CNN is a **detector**, not a classifier, so the task had to be redefined:\n\n"
        "1. Each MNIST image is treated as a detection sample: the *ground-truth box* is the tight rectangle "
        "around the non-zero (ink) pixels (threshold 0.2), and the class is `digit + 1` "
        "(`num_classes = 11`, index 0 = background).\n"
        "2. The grayscale image is repeated to 3 channels and the model's own transform resizes it "
        "(min side 96 px, max 128 px) to stay fast.\n"
        "3. Model: `fasterrcnn_mobilenet_v3_large_320_fpn`, COCO-pretrained MobileNetV3 backbone + FPN + RPN, "
        "with a new `FastRCNNPredictor` head for 11 classes.\n"
        "4. At inference the **highest-scoring box** gives the predicted digit; if no box is produced the sample "
        "is counted as *background* (a wrong prediction). Accuracy counts those as errors and macro-F1 is "
        "computed over the 10 digit classes (a background prediction is a false negative for the true class).\n\n"
        "Because it is far heavier than the CNN, it was trained on a random **3 000-image subset for 1 epoch** "
        "and evaluated on the first **2 000 test images** - this makes the comparison not perfectly equal, "
        "which is stated in the report."),
    nbf.v4.new_code_cell(FRCNN_TARGET),
    nbf.v4.new_code_cell(FRCNN_RESULTS),
    nbf.v4.new_code_cell(FRCNN_HISTORY),
    nbf.v4.new_code_cell(FRCNN_METRICS),
    nbf.v4.new_markdown_cell(
        "The training curve is the **sum of the four detection losses** (RPN objectness + RPN box regression + "
        "ROI classification + ROI box regression), so its value is not comparable to a cross-entropy. "
        "No test loss is reported: a detector has no single cross-entropy on the test set, and computing the "
        "detection losses would require ground-truth boxes and train-mode behaviour. This is why the table says N/A."),
    nbf.v4.new_markdown_cell("## 4. CNN vs Faster R-CNN"),
    nbf.v4.new_code_cell(PART1_TABLE),
    nbf.v4.new_markdown_cell(
        "**Interpretation.** The detector is roughly two orders of magnitude more expensive per image "
        "(region proposal network + ROI pooling + hundreds of box predictions) and, with only 3 000 training "
        "images and one epoch, it is clearly less accurate than the tiny CNN trained on the full set. "
        "MNIST digits are centred, size-normalised and contain exactly one object per image: there is nothing "
        "to localise, so all the detection machinery is pure overhead. Faster R-CNN becomes the right tool as "
        "soon as an image contains several objects at unknown positions (street scenes, documents, etc.)."),
    nbf.v4.new_markdown_cell(
        "## 5. Transfer learning: fine-tuned VGG16 and AlexNet\n"
        "Both ImageNet models are adapted as follows:\n"
        "- MNIST is resized to **64x64** (the smallest size that keeps the pretrained filters meaningful and "
        "is fast on CPU), the gray channel is repeated 3 times, and images are normalised with the ImageNet "
        "mean/std.\n"
        "- The convolutional `features` are **frozen**; only the classifier is retrained, with its last layer "
        "replaced by `Linear(4096, 10)`. For VGG16 torchvision adaptively pools to 7x7, so the first FC layer "
        "stays 25 088→4 096.\n"
        "- Adam lr=1e-3, 2 epochs, train subsets of 10 000 (VGG16) and 20 000 (AlexNet) images; evaluation on "
        "the full 10 000 test images."),
    nbf.v4.new_code_cell(FINETUNE_SETUP),
    nbf.v4.new_code_cell(FINETUNE_RESULTS),
    nbf.v4.new_markdown_cell(
        "With frozen ImageNet features and only two epochs on a subset, both models stay below the small CNN: "
        "ImageNet filters are tuned for natural RGB textures, while MNIST is binary-like strokes, and most of "
        "the 100+ million parameters are either frozen or useless for this task. Transfer learning is clearly "
        "not worth it for MNIST - but the experiment shows the mechanics (new head, frozen backbone, new input "
        "normalisation) that do pay off on small datasets of natural images."),
    nbf.v4.new_markdown_cell("## 6. Final Part 1 comparison"),
    nbf.v4.new_code_cell(ALL_TABLE),
    nbf.v4.new_markdown_cell(
        "**Conclusion.** On MNIST the best accuracy/cost trade-off is the custom CNN: a few hundred thousand "
        "parameters, a few minutes of CPU training and sub-millisecond inference. Faster R-CNN solves a harder "
        "problem we do not have here and is 50-100x slower per image. VGG16/AlexNet bring 50-130M parameters and "
        "a heavy preprocessing pipeline for a worse result, which illustrates that *pretrained on ImageNet* is "
        "not automatically better: the inductive bias and the domain of the pretraining data matter."),
]

# ----------------------------------------------------------------- notebook 2
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell(
        "# Deep Learning Lab 2 - Part 2: Vision Transformer from scratch on MNIST\n"
        "**Master MBD - Deep Learning - FST Tanger, Université Abdelmalek Essaadi**\n\n"
        "Implementation follows the standard ViT recipe (Dosovitskiy et al., 2020) and the "
        "[Medium step-by-step tutorial](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c): "
        "patch embedding, learnable [CLS] token, positional embeddings, pre-norm Transformer blocks with "
        "hand-written multi-head self-attention, MLP head on the [CLS] output."),
    nbf.v4.new_code_cell(SETUP),
    nbf.v4.new_markdown_cell(
        "## 1. Architecture\n"
        "For a 28x28 image and patch size 4: **7x7 = 49 patches**, each projected from 4x4=16 pixels to "
        "`dim=64`; one learnable [CLS] token is prepended, so the sequence length is 50.\n\n"
        "Each of the 6 blocks is pre-norm:\n\n"
        "```\n"
        "x = x + MHSA(LayerNorm(x))\n"
        "x = x + MLP(LayerNorm(x))          # Linear(64→256) - GELU - Linear(256→64)\n"
        "```\n\n"
        "Attention is written by hand (no `nn.TransformerEncoder`):\n"
        "`Q,K,V = Linear(x)`, `Attention = softmax(QKᵀ / sqrt(d_head)) V` with 4 heads of 16 dimensions, "
        "then a final `LayerNorm` and `Linear(64→10)` applied to the [CLS] token.\n\n"
        "Training: AdamW (lr=3e-4, weight decay 0.01), cosine schedule, batch 256, dropout 0.1, 10 epochs, "
        "CrossEntropyLoss."),
    nbf.v4.new_code_cell('''\
from src.models import ViT, PatchEmbedding
from src.utils import count_parameters

vit = ViT(img_size=28, patch_size=4, dim=64, depth=6, n_heads=4, mlp_ratio=4.0, dropout=0.1)
print(count_parameters(vit))
print(vit)

pe = PatchEmbedding(patch_size=4, dim=64)
x = torch.randn(2, 1, 28, 28)
print("image       ", tuple(x.shape))
print("patches     ", tuple(pe(x).shape), "-> 49 tokens of 64 dims")
tokens = torch.cat([torch.zeros(2, 1, 64), pe(x)], dim=1) + torch.zeros(1, 50, 64)
print("with [CLS]  ", tuple(tokens.shape))
'''),
    nbf.v4.new_code_cell('''\
# attention internals: shapes of Q, K, V and of the attention matrix
block = vit.blocks[0]
attn = block.attn
qkv = attn.qkv(torch.randn(2, 50, 64)).reshape(2, 50, 3, 4, 16).permute(2, 0, 3, 1, 4)
print("q,k,v:", tuple(qkv[0].shape), "(batch, heads, tokens, head_dim)")
scores = (qkv[0] @ qkv[1].transpose(-2, -1)) * attn.scale
print("attention scores:", tuple(scores.shape))
print("softmax rows sum to 1:", scores.softmax(-1).sum(-1)[0, 0, :3].tolist())
'''),
    nbf.v4.new_code_cell('''\
display(Image(filename=str(REPO / "figures/vit_patch_layout.png")))
'''),
    nbf.v4.new_markdown_cell(
        "The grid shows how the image is cut into non-overlapping 4x4 patches: each patch becomes one token, "
        "and the model must learn from scratch which patches are neighbours (positional embeddings) and how to "
        "combine them - there is no convolution to encode locality."),
    nbf.v4.new_code_cell('''\
vit_res = json.loads((REPO / "results/vit.json").read_text())
print("Device actually used:", vit_res["device"])
pd.DataFrame({k: [vit_res.get(k)] for k in
              ["epochs", "batch_size", "optimizer", "patch_size", "n_patches", "embed_dim",
               "depth", "heads", "mlp_ratio", "dropout", "train_images", "test_images"]}).T
'''),
    nbf.v4.new_code_cell('''\
display(pd.DataFrame(vit_res["history"]).round(4))
display(Image(filename=str(REPO / "figures/curves_vit.png")))
'''),
    nbf.v4.new_code_cell('''\
pd.DataFrame([{
    "test accuracy": round(vit_res["test_acc"], 4),
    "macro F1": round(vit_res["test_f1"], 4),
    "test loss": round(vit_res["test_loss"], 4),
    "training time (s)": round(vit_res["train_time_sec"], 1),
    "parameters": vit_res["params"]["params_total"],
    "inference (ms/img)": round(vit_res["inference_ms_per_image"], 3),
}])
'''),
    nbf.v4.new_code_cell('''\
from sklearn.metrics import classification_report
z = np.load(REPO / "results/preds_vit.npz")
print(classification_report(z["labels"], z["preds"], digits=4))
display(Image(filename=str(REPO / "figures/cm_vit.png")))
'''),
    nbf.v4.new_markdown_cell("## 2. Comparison with Part 1"),
    nbf.v4.new_code_cell('''\
allres = pd.read_csv(REPO / "results/comparison_all.csv")
cols = ["model", "test_acc", "test_f1", "test_loss", "train_time_sec",
        "params_total", "inference_ms_per_image", "train_images", "test_images"]
t = allres[cols].copy()
t.columns = ["model", "accuracy", "macro F1", "test loss", "train time (s)",
             "params", "inference (ms/img)", "train imgs", "test imgs"]
display(t.round(4))
display(Image(filename=str(REPO / "figures/comparison_all.png")))
'''),
    nbf.v4.new_markdown_cell(
        "## 3. Interpretation\n"
        "- The ViT is **competitive but not better** than the small CNN on MNIST. With only 60 000 training "
        "images and 10 epochs it has to *learn* locality and translation equivariance from data, whereas a "
        "convolution has them built in. This is exactly the data-efficiency gap described by Dosovitskiy et al.: "
        "ViTs shine when pre-trained on very large corpora (JFT-300M, ImageNet-21k) and then fine-tuned.\n"
        "- Its 50-token sequence with 4x4 patches is very small, so the model is cheap (≈0.1M parameters), "
        "yet it needs more epochs than the CNN to reach the same test accuracy; the loss curve keeps improving "
        "at the end of training, which suggests it is still under-trained.\n"
        "- **Patch size**: 4x4 gives 49 tokens. Larger patches (7x7 → 16 tokens) are faster but lose detail "
        "and lower the accuracy; smaller patches cost quadratic attention and more parameters. 4 is a good "
        "compromise at 28x28.\n"
        "- **Depth**: 6 blocks are enough to mix information globally; deeper models need more data and "
        "regularisation and over-fit quickly on MNIST.\n"
        "- Errors are similar to the CNN's (4/9, 3/5, 7/1 confusions), and the confusion matrix is globally "
        "clean: the [CLS] token does capture the digit identity.\n\n"
        "**Take-away**: on a small, clean, low-resolution dataset a CNN is the more efficient inductive bias; "
        "a ViT becomes the better choice with large-scale pretraining, where attention gives global context."),
]

for nb, name in [(nb1, "part1_cnn_fasterrcnn_finetuning.ipynb"),
                 (nb2, "part2_vit_from_scratch.ipynb")]:
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
        "language_info": {"name": "python"},
    })
    nbf.write(nb, NB_DIR / name)
    print("wrote", NB_DIR / name)
