import numpy as np, pandas as pd, os, time
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

OUT='workspaces/sirius_tau'
CAND_CPATH=f'{OUT}/crops/crops_ground.npy'
FULL_CPATH=f'{OUT}/crops/crops_ground_all.npy'
TEST_CPATH=f'{OUT}/crops/crops_ground_test.npy'
T=7; HH=WW=64
BS=int(os.environ.get('BS','512'))
EPOCHS=int(os.environ.get('EPOCHS','8'))
SEED_OFF=int(os.environ.get('SEED_OFF','0'))
OOFN=os.environ.get('OOFN',f'{OUT}/cnn_ground_oof_all.npy')
TESTN=os.environ.get('TESTN',f'{OUT}/cnn_ground_test_all.npy')
LR=float(os.environ.get('LR','2e-3'))
NF=5
MODELDIR=f'{OUT}/models_ground'; os.makedirs(MODELDIR,exist_ok=True)
torch.backends.cudnn.benchmark=True

class DS(Dataset):
    def __init__(self,rows,y,mm,train): self.rows=rows; self.y=y; self.mm=mm; self.train=train
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        x=np.asarray(self.mm[r]).reshape(T*2*3,HH,WW).astype(np.float32)/255.0
        if self.train and np.random.rand()<0.5: x=x[:,:,::-1].copy()
        return torch.from_numpy(x), torch.tensor(self.y[r])

class Net(nn.Module):
    def __init__(self,inc=T*2*3):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.MaxPool2d(2))
        self.b=nn.Sequential(blk(inc,32),blk(32,64),blk(64,128),blk(128,128),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(self,x): return self.head(self.b(x)).squeeze(1)

def predict(net,rows,mm,bs=2048):
    net.eval(); out=np.zeros(len(rows))
    with torch.no_grad():
        for i in range(0,len(rows),bs):
            rr=rows[i:i+bs]
            xb=np.stack([np.asarray(mm[r]).reshape(T*2*3,HH,WW).astype(np.float32)/255.0 for r in rr])
            out[i:i+len(rr)]=net(torch.from_numpy(xb).cuda()).cpu().numpy()
    return out

def main():
    full=pd.read_parquet(f'{OUT}/ground_all_index.parquet')
    cand=pd.read_parquet(f'{OUT}/ground_index.parquet')
    yfull=full.contact.values.astype(np.float32)
    ycand=cand.contact.values.astype(np.float32)
    key_full=(full.game_play+'_'+full.step.astype(str)+'_'+full.nfl_player_id_1.astype(str)).values
    key_cand=(cand.game_play+'_'+cand.step.astype(str)+'_'+cand.nfl_player_id_1.astype(str)).values
    fold_full=np.zeros(len(full),int)
    for k,(tri,vai) in enumerate(GroupKFold(n_splits=NF).split(full,yfull,full.game_play.values)):
        fold_full[vai]=k
    kf=dict(zip(key_full,fold_full))
    fold_cand=np.array([kf[x] for x in key_cand])
    MM_C=np.load(CAND_CPATH,mmap_mode='r')
    MM_F=np.load(FULL_CPATH,mmap_mode='r')
    oof=np.full(len(full),np.nan)
    for k in range(NF):
        tr=np.where(fold_cand!=k)[0]
        va=np.where(fold_full==k)[0]
        t0=time.time(); print(f'fold {k} ntrain {len(tr)} nval {len(va)}',flush=True)
        torch.manual_seed(k+SEED_OFF); np.random.seed(k+SEED_OFF)
        dl=DataLoader(DS(tr,ycand,MM_C,True),batch_size=BS,shuffle=True,num_workers=6,drop_last=True)
        net=Net().cuda(); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=EPOCHS*len(dl),pct_start=0.2)
        lossf=nn.BCEWithLogitsLoss()
        for ep in range(EPOCHS):
            net.train(); tl=0
            for xb,yb in dl:
                xb=xb.cuda(); yb=yb.cuda(); opt.zero_grad()
                loss=lossf(net(xb),yb); loss.backward(); opt.step(); sched.step(); tl+=loss.item()
            if ep==EPOCHS-1:
                pv=predict(net,va,MM_F); print(f'  ep{ep} loss {tl/len(dl):.4f} valAUC {roc_auc_score(yfull[va],pv):.4f} t {time.time()-t0:.0f}',flush=True)
        oof[va]=predict(net,va,MM_F)
        torch.save(net.state_dict(),f'{MODELDIR}/fold{k}.pt')
        np.save(OOFN,oof)
        print('fold done',k,'auc',roc_auc_score(yfull[va],oof[va]),flush=True)
    m=~np.isnan(oof); print('OOF all AUC',roc_auc_score(yfull[m],oof[m]),'n',int(m.sum()),flush=True)
    # test ensemble
    ti=pd.read_parquet(f'{OUT}/ground_test_index.parquet'); TM=np.load(TEST_CPATH,mmap_mode='r')
    rows=list(range(len(ti))); P=np.zeros(len(rows))
    for k in range(NF):
        net=Net().cuda(); net.load_state_dict(torch.load(f'{MODELDIR}/fold{k}.pt')); P+=predict(net,rows,TM)
    P/=NF; np.save(TESTN,P); print('TEST saved mean',P.mean(),flush=True)

if __name__=='__main__': main()
