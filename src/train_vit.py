import json, time, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(str(ROOT / "data"), train=True, download=True, transform=transform)
test_ds = datasets.MNIST(str(ROOT / "data"), train=False, download=True, transform=transform)
# smaller for CPU if needed
tr_idx = torch.randperm(len(train_ds))[:20000]
train_loader = DataLoader(Subset(train_ds,tr_idx), batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(DataLoader(test_ds,batch_size=128,num_workers=0).dataset, batch_size=128, num_workers=0)  # full test

class PatchEmbed(nn.Module):
    def __init__(self,img=28,patch=7,dim=64):
        super().__init__()
        self.p = (img//patch)**2
        self.proj = nn.Conv2d(1,dim,kernel_size=patch,stride=patch)
    def forward(self,x):
        x=self.proj(x).flatten(2).transpose(1,2); return x

class Attention(nn.Module):
    def __init__(self,dim=64,heads=4):
        super().__init__()
        self.h=heads; self.d=dim//heads; self.scale=self.d**-0.5
        self.qkv=nn.Linear(dim,dim*3); self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.h,self.d).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        att=(q@k.transpose(-2,-1))*self.scale; att=att.softmax(-1)
        x=(att@v).transpose(1,2).reshape(B,N,C); return self.proj(x)

class ViT(nn.Module):
    def __init__(self,img=28,patch=7,dim=64,depth=4,heads=4,mlp=2):
        super().__init__()
        pe=PatchEmbed(img,patch,dim)
        self.p=pe.p; self.embed=pe
        self.cls=nn.Parameter(torch.zeros(1,1,dim))
        self.pos=nn.Parameter(torch.zeros(1,1+self.p,dim))
        self.blocks=nn.ModuleList([nn.ModuleDict({
            'n1':nn.LayerNorm(dim),'att':Attention(dim,heads),'n2':nn.LayerNorm(dim),
            'mlp':nn.Sequential(nn.Linear(dim,dim*mlp),nn.GELU(),nn.Linear(dim*mlp,dim))
        }) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,10)
    def forward(self,x):
        x=self.embed(x); B=x.shape[0]
        x=torch.cat([self.cls.expand(B,-1,-1),x],1)+self.pos
        for b in self.blocks:
            x=x+b['att'](b['n1'](x)); x=x+b['mlp'](b['n2'](x))
        return self.head(self.norm(x)[:,0])

model = ViT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
crit = nn.CrossEntropyLoss()
hist=[]; t0=time.time()
for ep in range(1,4):  # 3 epochs
    model.train(); tl,tc,tn=0,0,0
    for i,(x,y) in enumerate(train_loader):
        x,y=x.to(device),y.to(device)
        opt.zero_grad(); out=model(x); loss=crit(out,y); loss.backward(); opt.step()
        tl+=loss.item()*y.size(0); tc+=(out.argmax(1)==y).sum().item(); tn+=y.size(0)
        if i%100==0: print(f"vit ep{ep} {i}/{len(train_loader)} loss={loss.item():.3f}")
    model.eval(); preds,labels=[],[]
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device); out=model(x); preds+=out.argmax(1).cpu().tolist(); labels+=y.cpu().tolist()
    hist.append({"epoch":ep,"train_acc":tc/tn,"test_acc":accuracy_score(labels,preds),"train_loss":tl/tn})
    print(f"vit ep{ep} ta={tc/tn:.4f} va={accuracy_score(labels,preds):.4f}")
tr=time.time()-t0
res={"model":"ViT","device":str(device),"epochs":3,"batch_size":128,"optimizer":"AdamW(lr=3e-4)","train_images":20000,"test_images":10000,
     "params_total":sum(p.numel() for p in model.parameters()),"train_time_sec":tr,
     "test_loss":None,"test_acc":accuracy_score(labels,preds),"test_f1":f1_score(labels,preds,average='macro'),
     "inference_ms_per_image":0,"history":hist,"notes":"patch7,dim64,heads4,depth4,cls+pos"}
json.dump(res, open(RES/"vit.json",'w'), indent=2)
print("saved vit")
