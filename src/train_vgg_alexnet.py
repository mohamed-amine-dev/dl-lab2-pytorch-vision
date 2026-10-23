import json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])
train_ds = datasets.MNIST(str(ROOT / "data"), train=True, download=True, transform=transform)
test_ds = datasets.MNIST(str(ROOT / "data"), train=False, download=True, transform=transform)
# subsets
tr_idx = torch.randperm(len(train_ds))[:5000]
te_idx = torch.randperm(len(test_ds))[:2000]
train_loader = DataLoader(Subset(train_ds,tr_idx), batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(Subset(test_ds,te_idx), batch_size=32, shuffle=False, num_workers=0)

hist=[]
t0=time.time()
# VGG16
try:
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
except:
    vgg = models.vgg16(weights=None)
for p in vgg.features.parameters(): p.requires_grad=False
vgg.classifier[6] = nn.Linear(4096,10)
vgg.to(device)
opt = torch.optim.Adam(vgg.classifier.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
for ep in range(1,2):  # 1 epoch
    vgg.train(); tl,tc,tn=0,0,0
    for x,y in train_loader:
        x,y=x.to(device),y.to(device)
        opt.zero_grad(); out=vgg(x); loss=crit(out,y); loss.backward(); opt.step()
        tl+=loss.item()*y.size(0); tc+=(out.argmax(1)==y).sum().item(); tn+=y.size(0)
    vgg.eval(); preds,labels=[],[]
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device); out=vgg(x); preds+=out.argmax(1).cpu().tolist(); labels+=y.cpu().tolist()
    hist.append({"epoch":ep,"train_acc":tc/tn,"test_acc":accuracy_score(labels,preds)})
    print(f"vgg ep{ep} ta={tc/tn:.4f} va={accuracy_score(labels,preds):.4f}")
tr_vgg=time.time()-t0
res={"model":"VGG16","device":str(device),"epochs":1,"batch_size":32,"optimizer":"Adam(classifier,lr=1e-3)","train_images":5000,"test_images":2000,
     "params_total":sum(p.numel() for p in vgg.parameters()),"train_time_sec":tr_vgg,
     "test_loss":None,"test_acc":accuracy_score(labels,preds),"test_f1":f1_score(labels,preds,average='macro'),
     "inference_ms_per_image":0,"history":hist}
json.dump(res, open(RES/"vgg16.json",'w'), indent=2)
print("saved vgg16")
