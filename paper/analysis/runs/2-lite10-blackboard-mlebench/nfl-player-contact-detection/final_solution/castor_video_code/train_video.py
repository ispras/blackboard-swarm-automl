import os, sys, time, math, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import roc_auc_score, matthews_corrcoef
R="/download_job_code/payload/sandbox"; W=f"{R}/workspaces/castor_alpha"
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)

class CropDS(Dataset):
    def __init__(self, df, crops, tab, train=False):
        self.df=df.reset_index(drop=True); self.crops=crops; self.tab=tab; self.train=train
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r=self.df.iloc[i]; sid=int(r.sample_id)
        a=self.crops[sid,0]; b=self.crops[sid,1]
        if self.train:
            if np.random.rand()<0.5: a=a[:,::-1]
            if np.random.rand()<0.5: b=b[:,::-1]
            if np.random.rand()<0.5:
                f=np.random.uniform(0.85,1.15); a=np.clip(a.astype(np.float32)*f,0,255).astype(np.uint8); b=np.clip(b.astype(np.float32)*f,0,255).astype(np.uint8)
        a=a[...,::-1].astype(np.float32)/255.0; b=b[...,::-1].astype(np.float32)/255.0
        a=(a-MEAN)/STD; b=(b-MEAN)/STD
        a=torch.from_numpy(a.transpose(2,0,1)); b=torch.from_numpy(b.transpose(2,0,1))
        t=torch.from_numpy(self.tab[i])
        y=torch.tensor(float(r.y))
        return a,b,t,y

class Net(nn.Module):
    def __init__(self, ntab, arch="resnet18"):
        super().__init__()
        self.enc=timm.create_model(arch,pretrained=True,num_classes=0)
        d=self.enc.num_features
        self.head=nn.Sequential(nn.Linear(d*3+ntab,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,1))
    def forward(self,a,b,t):
        ea=self.enc(a); eb=self.enc(b)
        x=torch.cat([ea,eb,torch.abs(ea-eb),t],dim=1)
        return self.head(x).squeeze(1)

def prep_tab(meta, kind, split):
    fp=f"{W}/{kind}_{split}_feats.parquet"
    if not os.path.exists(fp):
        print("no tab feats",fp,flush=True); return np.zeros((len(meta),0),np.float32),[]
    f=pd.read_parquet(fp)
    drop={"contact_id","game_play","pid","step","y","fold","pid2","sample_id"}
    cols=[c for c in f.columns if c not in drop and pd.api.types.is_numeric_dtype(f[c])]
    f=f[["contact_id"]+cols]
    m=meta[["contact_id"]].merge(f,on="contact_id",how="left")
    X=m[cols].to_numpy(np.float32)
    return X,cols

def main(kind="ground", epochs=6, only_fold=None, bs=128, lr=3e-4, arch="resnet18"):
    meta=pd.read_parquet(f"{W}/crops/{kind}_train_meta.parquet")
    crops=np.load(f"{W}/crops/{kind}_train.npy",mmap_mode="r")
    X,cols=prep_tab(meta,kind,"train")
    # impute + standardize
    if X.shape[1]>0:
        med=np.nanmedian(X,axis=0); med=np.where(np.isfinite(med),med,0).astype(np.float32)
        X=np.where(np.isfinite(X),X,med).astype(np.float32)
        mu=X.mean(0); sd=X.std(0)+1e-6
        X=(X-mu)/sd
        np.save(f"{W}/tabstats_{kind}.npy",{"med":med,"mu":mu,"sd":sd,"cols":cols},allow_pickle=True)
    print("kind",kind,"N",len(meta),"ntab",X.shape[1],flush=True)
    device="cuda"
    oof=np.full(len(meta),np.nan); folds=sorted(meta.fold.unique())
    for f in folds:
        if only_fold is not None and f!=only_fold: continue
        tr=meta.fold!=f; va=meta.fold==f
        # class balance on train
        ytr=meta.loc[tr,"y"].to_numpy(); pw=float((ytr==0).sum())/max(1,(ytr==1).sum()); pw=min(pw,8.0)
        dl_tr=DataLoader(CropDS(meta[tr],crops,X[tr],True),batch_size=bs,shuffle=True,num_workers=8,drop_last=True,persistent_workers=True,pin_memory=True)
        dl_va=DataLoader(CropDS(meta[va],crops,X[va],False),batch_size=256,shuffle=False,num_workers=6,persistent_workers=True,pin_memory=True)
        model=Net(X.shape[1],arch).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
        lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=device))
        best=(-1,None,0)
        for ep in range(epochs):
            model.train(); t0=time.time(); tot=0
            for a,b,t,y in dl_tr:
                a=a.to(device,non_blocking=True); b=b.to(device,non_blocking=True); t=t.to(device); y=y.to(device)
                opt.zero_grad(); out=model(a,b,t); loss=lossf(out,y); loss.backward(); opt.step(); tot+=loss.item()*len(y)
            sched.step()
            model.eval(); pv=[]
            with torch.no_grad():
                for a,b,t,y in dl_va:
                    a=a.to(device); b=b.to(device); t=t.to(device)
                    pv.append(torch.sigmoid(model(a,b,t)).cpu().numpy())
            pv=np.concatenate(pv); auc=roc_auc_score(meta.loc[va,"y"],pv)
            print(f"fold{f} ep{ep} loss{tot/tr.sum():.4f} auc{auc:.4f} {time.time()-t0:.1f}s",flush=True)
            if auc>best[0]: best=(auc,{k:v.detach().cpu() for k,v in model.state_dict().items()},ep)
        model.load_state_dict(best[1]); torch.save({"state":best[1],"ntab":X.shape[1],"cols":cols,"auc":best[0],"arch":arch},f"{W}/models/{kind}_{arch}_f{f}.pt")
        # recompute val preds with best
        model.eval(); pv=[]
        with torch.no_grad():
            for a,b,t,y in dl_va:
                a=a.to(device); b=b.to(device); t=t.to(device); pv.append(torch.sigmoid(model(a,b,t)).cpu().numpy())
        pv=np.concatenate(pv); oof[va.to_numpy()]=pv
        print(f"FOLD{f} best_auc {best[0]:.4f} ep{best[2]}",flush=True)
    pd.DataFrame({"contact_id":meta.contact_id,"y":meta.y,"fold":meta.fold,"oof":oof}).to_parquet(f"{W}/oof_{kind}_vid_subset.parquet",index=False)
    m=~np.isnan(oof)
    print("SUBSET OOF AUC",roc_auc_score(meta.y[m],oof[m]),flush=True)
    best=(0,0.5)
    for th in np.unique(np.quantile(oof[m],np.linspace(0.5,0.999,400))):
        mm=matthews_corrcoef(meta.y[m],(oof[m]>=th).astype(int))
        if mm>best[0]: best=(mm,th)
    print("SUBSET OOF MCC",best,flush=True)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--kind",default="ground"); ap.add_argument("--epochs",type=int,default=6)
    ap.add_argument("--fold",type=int,default=-1); ap.add_argument("--bs",type=int,default=128); ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--arch",default="resnet18")
    a=ap.parse_args(); main(a.kind,a.epochs,None if a.fold<0 else a.fold,a.bs,a.lr,a.arch)
