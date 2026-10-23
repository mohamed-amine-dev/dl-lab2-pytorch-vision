import json, time, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(str(ROOT / "data"), train=True, download=True, transform=transform)
test_ds = datasets.MNIST(str(ROOT / "data"), train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7,128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128,10)
        )
    def forward(self,x):
        return self.classifier(self.features(x))

model = CNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
hist = []
t0=time.time()
for ep in range(1,6):
    model.train()
    tl,tc,tn=0,0,0
    for x,y in train_loader:
        x,y=x.to(device),y.to(device)
        opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); opt.step()
        tl+=loss.item()*y.size(0); tc+=(out.argmax(1)==y).sum().item(); tn+=y.size(0)
    model.eval(); vl,vc,vn=0,0,0; preds,labels=[],[]
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device)
            out=model(x); loss=crit(out,y)
            vl+=loss.item()*y.size(0); vc+=(out.argmax(1)==y).sum().item(); vn+=y.size(0)
            preds+=out.argmax(1).cpu().tolist(); labels+=y.cpu().tolist()
    hist.append({"epoch":ep,"train_loss":tl/tn,"train_acc":tc/tn,"test_loss":vl/vn,"test_acc":vc/vn})
    print(f"ep{ep} tl={tl/tn:.4f} ta={tc/tn:.4f} vl={vl/vn:.4f} va={vc/vn:.4f}")
tr=time.time()-t0
res={"model":"CNN","device":str(device),"epochs":5,"batch_size":256,"optimizer":"Adam(lr=1e-3,weight_decay=1e-4)","train_images":60000,"test_images":10000,
     "params_total":sum(p.numel() for p in model.parameters()),"train_time_sec":tr,
     "test_loss":hist[-1]["test_loss"],"test_acc":hist[-1]["test_acc"],"test_f1":f1_score(labels,preds,average='macro'),
     "inference_ms_per_image":0,"history":hist}
json.dump(res, open(RES/"cnn.json",'w'), indent=2)
print("saved cnn")
