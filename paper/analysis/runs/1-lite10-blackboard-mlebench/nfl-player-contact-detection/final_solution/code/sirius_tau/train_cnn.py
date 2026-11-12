import numpy as np, pandas as pd, os, sys, time, json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

OUT='workspaces/sirius_tau'
CPATH=os.environ.get('CROPS',f'{OUT}/crops/crops_train.npy')
IDXP=os.environ.get('IDX',f'{OUT}/crops_index.parquet')
OOFN=os.environ.get('OOFN',f'{OUT}/cnn_oof.npy')
T=int(os.environ.get('T','5'))
HH=WW=64
BS=int(os.environ.get('BS','512'))
EPOCHS=int(os.environ.get('EPOCHS','10'))
LR=float(os.environ.get('LR','2e-3'))
MAXF=int(os.environ.get('MAXF','5'))
torch.backends.cudnn.benchmark=True

idx=pd.read_parquet(IDXP)
N=len(idx)
Y=idx.contact.values.astype(np.float32)
GROUPS=idx.game_play.values
MM=np.load(CPATH, mmap_mode='r')

class DS(Dataset):
    def __init__(self, rows, train):
        self.rows=rows; self.train=train
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        x=np.asarray(MM[r]).reshape(T*2*3,HH,WW).astype(np.float32)/255.0
        if self.train and np.random.rand()<0.5:
            x=x[:,:,::-1].copy()
        return torch.from_numpy(x), torch.tensor(Y[r])

class Net(nn.Module):
    def __init__(self, inc=30):
        super().__init__()
        def blk(i,o):
            return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.b=nn.Sequential(blk(inc,32),blk(32,64),blk(64,128),blk(128,128),
                             nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(self,x): return self.head(self.b(x)).squeeze(1)

def predict(net,rows,bs=2048):
    net.eval(); out=np.zeros(len(rows))
    with torch.no_grad():
        for i in range(0,len(rows),bs):
            rr=rows[i:i+bs]
            xb=np.stack([np.asarray(MM[r]).reshape(T*2*3,HH,WW).astype(np.float32)/255.0 for r in rr])
            out[i:i+len(rr)]=net(torch.from_numpy(xb).cuda()).cpu().numpy()
    return out

def run_fold(tri,vai,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dl=DataLoader(DS(tri,True),batch_size=BS,shuffle=True,num_workers=6,drop_last=True)
    net=Net(T*2*3).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=EPOCHS*len(dl),pct_start=0.2)
    lossf=nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        net.train(); tl=0; t0=time.time()
        for xb,yb in dl:
            xb=xb.cuda(); yb=yb.cuda()
            opt.zero_grad(); out=net(xb); loss=lossf(out,yb); loss.backward()
            opt.step(); sched.step(); tl+=loss.item()
        pv=predict(net,vai)
        print(f'  ep{ep} loss {tl/len(dl):.4f} valAUC {roc_auc_score(Y[vai],pv):.4f} t {time.time()-t0:.0f}',flush=True)
    return predict(net,vai)

def main():
    gkf=GroupKFold(n_splits=5); oof=np.zeros(N); done=0
    for k,(tri,vai) in enumerate(gkf.split(np.zeros(N),Y,GROUPS)):
        if k>=MAXF: break
        t0=time.time(); print('fold',k,'ntr',len(tri),'nva',len(vai),flush=True)
        oof[vai]=run_fold(tri,vai,seed=k)
        np.save(OOFN,oof)
        print('fold done',k,'auc',roc_auc_score(Y[vai],oof[vai]),'t',round(time.time()-t0),flush=True)
    np.save(OOFN,oof)
    m=oof!=0
    if m.sum()>0: print('partial OOF AUC',roc_auc_score(Y[m],oof[m]),'n',m.sum(),flush=True)

if __name__=='__main__': main()
