import json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# MNIST as detection
ds_train = datasets.MNIST(str(ROOT / "data"), train=True, download=True)
ds_test = datasets.MNIST(str(ROOT / "data"), train=False, download=True)

class MNISTDet(torch.utils.data.Dataset):
    def __init__(self, base, count):
        self.base = base
        self.idx = np.random.choice(len(base), count, replace=False)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        img, label = self.base[self.idx[i]]
        img = transforms.ToTensor()(img)  # 1,28,28
        img3 = img.repeat(3,1,1)
        # box from nonzero
        arr = (img[0]>0.1).nonzero(as_tuple=False)
        if len(arr)==0:
            box = torch.tensor([[0,0,27,27]], dtype=torch.float32)
        else:
            y1,x1 = arr.min(0)[0]; y2,x2 = arr.max(0)[0]
            box = torch.tensor([[float(x1),float(y1),float(x2+1),float(y2+1)]], dtype=torch.float32)
        target = {"boxes":box, "labels":torch.tensor([int(label)+1])}
        return img3, target

def collate(b): return tuple(zip(*b))

train_ds = MNISTDet(ds_train, 2000)
test_ds = MNISTDet(ds_test, 1000)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate, num_workers=0)

try:
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
except:
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, 11)
model.transform.min_size = (64,); model.transform.max_size = 64
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
hist=[]
t0=time.time()
for ep in range(1,2):  # 1 epoch
    model.train()
    total=0
    for i,(imgs,targets) in enumerate(train_loader):
        imgs=[x.to(device) for x in imgs]
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(v for v in loss_dict.values())
        opt.zero_grad(); loss.backward(); opt.step()
        total+=loss.item()
    hist.append({"epoch":ep,"train_loss":total/len(train_loader)})
    print(f"ep{ep} loss={total/len(train_loader):.4f}")
tr=time.time()-t0

model.eval(); preds,labels=[],[]
with torch.no_grad():
    for imgs,targets in test_loader:
        imgs=[x.to(device) for x in imgs]
        out = model(imgs)
        for o,t in zip(out,targets):
            if len(o['scores'])==0: preds.append(10)
            else:
                idx = o['scores'].argmax().item(); preds.append(int(o['labels'][idx].item())-1)
            labels.append(int(t['labels'][0].item())-1)
# ignore background in f1? use all 10
preds_arr = np.array([p if p<10 else 0 for p in preds]) # treat bg as 0 (conservative)
labels_arr = np.array(labels)
acc = accuracy_score(labels_arr, np.array(preds))
f1 = f1_score(labels_arr, np.array(preds), average='macro')
res={"model":"Faster R-CNN","device":str(device),"epochs":1,"batch_size":4,"optimizer":"SGD(lr=0.005,momentum=0.9)","train_images":2000,"test_images":1000,
     "params_total":sum(p.numel() for p in model.parameters()),"train_time_sec":tr,
     "test_loss":None,"test_acc":float(acc),"test_f1":float(f1),
     "inference_ms_per_image":0,"history":hist,"notes":"boxes from ink pixels, 11 classes, highest box used"}
json.dump(res, open(RES/"frcnn.json",'w'), indent=2)
print("saved frcnn", acc, f1)
