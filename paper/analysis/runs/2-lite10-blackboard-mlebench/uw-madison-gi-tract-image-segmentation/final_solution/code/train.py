import os, sys, argparse, time, json, math, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
sys.path.insert(0,os.path.dirname(__file__))
from common import *

def get_folds(df, n=5, seed=42):
    board_f=f"{BASE}/blackboard/betelgeuse_zeta_folds.csv"
    if os.path.exists(board_f):
        fo=pd.read_csv(board_f)
        m=dict(zip(fo["group"],fo["fold"]))
        if df.case_day.isin(m.keys()).all():
            return df.case_day.map(m).values
    fpath=f"{WS}/folds_groupcd.csv"
    if os.path.exists(fpath):
        fo=pd.read_csv(fpath)
        if len(fo)==df.case_day.nunique():
            m=dict(zip(fo.case_day,fo.fold))
            return df.case_day.map(m).values
    gkf=GroupKFold(n_splits=n)
    folds=np.zeros(len(df),dtype=int)
    cds=df.case_day.values
    uniq=df.groupby("case_day").head(1).index.values
    X=np.zeros((len(uniq),1)); y=np.zeros(len(uniq)); groups=df.loc[uniq,"case_day"].values
    for k,(_,vi) in enumerate(gkf.split(X,y,groups)):
        for g in groups[vi]: pass
    # simpler: assign fold per case_day
    g=np.random.RandomState(seed)
    cds_sorted=sorted(df.case_day.unique())
    # group by case so same case's days spread? use GroupKFold on case to avoid patient leak? use cd
    fold_of={}
    from collections import defaultdict
    # GroupKFold by case to keep patients together is stricter; but test overlaps patients, so use case_day grouping
    gkf2=GroupKFold(n_splits=n)
    cd_list=df.groupby("case_day").head(1)[["case_day"]].reset_index(drop=True)
    for k,(_,vi) in enumerate(gkf2.split(np.zeros(len(cd_list)),np.zeros(len(cd_list)),cd_list.case_day.values)):
        for cd in cd_list.case_day.values[vi]: fold_of[cd]=k
    pd.DataFrame({"case_day":list(fold_of.keys()),"fold":list(fold_of.values())}).to_csv(fpath,index=False)
    return df.case_day.map(fold_of).values

class EMA:
    def __init__(self,m,decay=0.999):
        self.ema=m; self.decay=decay; self.shadow={k:v.detach().clone() for k,v in m.state_dict().items()}
    def update(self,m):
        for k,v in m.state_dict().items():
            if v.dtype.is_floating_point: self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
            else: self.shadow[k]=v.detach().clone()

def dice_torch(pred,gt,eps=1e-6):
    inter=(pred*gt).sum(dim=(0,2,3)); s=pred.sum(dim=(0,2,3))+gt.sum(dim=(0,2,3))
    return ((2*inter+eps)/(s+eps))

def validate(model, dl, device, thr=0.5):
    model.eval()
    inter=np.zeros(3); psum=np.zeros(3); gsum=np.zeros(3)
    per_pos=[]; fp=0; npos=0; nneg=0
    with torch.no_grad():
        for x,y in dl:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            with torch.cuda.amp.autocast():
                logit=model(x)
            p=(torch.sigmoid(logit)>thr).float()
            inter+=(p*y).sum(dim=(0,2,3)).cpu().numpy()
            psum+=p.sum(dim=(0,2,3)).cpu().numpy()
            gsum+=y.sum(dim=(0,2,3)).cpu().numpy()
            # per-slice-present dice
            ps=p.cpu().numpy(); ys=y.cpu().numpy()
            for c in range(3):
                g=ys[:,c]; pr=ps[:,c]
                pos=g.sum(axis=(1,2))>0
                if pos.any():
                    i=(pr[pos]*g[pos]).sum(axis=(1,2)); s=pr[pos].sum(axis=(1,2))+g[pos].sum(axis=(1,2))
                    d=np.where(s>0,(2*i)/(s+1e-6),1.0)
                    per_pos.append(d)
                neg=~pos
                if neg.any():
                    fp+=pr[neg].sum()
                    nneg+=neg.sum()
    dcls=(2*inter+1e-6)/(psum+gsum+1e-6)
    pos_dice=np.concatenate(per_pos).mean() if per_pos else 0.0
    # fp per negative slice per class
    fp_rate=fp/max(nneg,1)
    return dcls, pos_dice, fp_rate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold",type=int,default=0)
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--size",type=int,default=320)
    ap.add_argument("--encoder",default="resnet34")
    ap.add_argument("--arch",default="Unet")
    ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--name",default="r34_f0")
    ap.add_argument("--twofive",type=int,default=1)
    ap.add_argument("--max_train",type=int,default=0)
    ap.add_argument("--max_val",type=int,default=0)
    ap.add_argument("--all_data",type=int,default=0)
    ap.add_argument("--warmup_epochs",type=float,default=1.0)
    ap.add_argument("--clip",type=float,default=5.0)
    args=ap.parse_args()
    torch.manual_seed(42+args.fold); np.random.seed(42+args.fold)
    outdir=f"{WS}/runs/{args.name}"; os.makedirs(outdir,exist_ok=True)
    df=build_slice_df()
    folds=get_folds(df)
    df["fold"]=folds
    if args.all_data:
        tr_idx=np.arange(len(df)); va_idx=np.array([],dtype=int)
    else:
        tr_idx=np.where(df.fold!=args.fold)[0]; va_idx=np.where(df.fold==args.fold)[0]
    if args.max_train: tr_idx=tr_idx[:args.max_train]
    if args.max_val: va_idx=va_idx[:args.max_val]
    print(f"train {len(tr_idx)} val {len(va_idx)}",flush=True)
    tr_ds=GIDataset(df,size=args.size,train=True,indices=tr_idx,twofive=bool(args.twofive))
    tr_dl=DataLoader(tr_ds,batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True,persistent_workers=args.workers>0)
    va_dl=None
    if len(va_idx)>0:
        va_ds=GIDataset(df,size=args.size,train=False,indices=va_idx,twofive=bool(args.twofive))
        va_dl=DataLoader(va_ds,batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True)
    import segmentation_models_pytorch as smp
    model=getattr(smp,args.arch)(encoder_name=args.encoder,encoder_weights="imagenet",
                                 in_channels=3 if args.twofive else 1,classes=3)
    model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    steps=len(tr_dl)*args.epochs; warm_steps=int(len(tr_dl)*args.warmup_epochs)
    def lr_at(it):
        if it<warm_steps: return args.lr*it/max(1,warm_steps)
        p=(it-warm_steps)/max(1,steps-warm_steps)
        return args.lr*0.5*(1+math.cos(math.pi*p))
    glob_it=[0]
    bce=nn.BCEWithLogitsLoss(); dice=smp.losses.DiceLoss(mode="multilabel",from_logits=True)
    scaler=torch.cuda.amp.GradScaler()
    best=-1; log=[]
    for ep in range(args.epochs):
        model.train(); t0=time.time(); tot=0; n=0
        for x,y in tr_dl:
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lr=lr_at(glob_it[0]); glob_it[0]+=1
            for g in opt.param_groups: g["lr"]=lr
            with torch.cuda.amp.autocast():
                logit=model(x); loss=bce(logit,y)+dice(logit,y)
            scaler.scale(loss).backward()
            if args.clip>0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
            scaler.step(opt); scaler.update()
            tot+=loss.item(); n+=1
        if va_dl is not None:
            dcls,posd,fpr=validate(model,va_dl, torch.device("cuda"))
        else:
            dcls=np.zeros(3); posd=0.0; fpr=0.0
        rec=dict(epoch=ep,loss=float(tot/max(n,1)),dice_lb=float(dcls[0]),dice_sb=float(dcls[1]),dice_st=float(dcls[2]),
                 dice_pos=float(posd),fp_rate=float(fpr),lr=float(lr),sec=float(time.time()-t0))
        log.append(rec); print(json.dumps(rec),flush=True)
        pd.DataFrame(log).to_csv(f"{outdir}/log.csv",index=False)
        if posd>best:
            best=posd
            torch.save({"state":model.state_dict(),"args":vars(args),"best":best},f"{outdir}/best.pt")
        torch.save({"state":model.state_dict(),"args":vars(args),"best":best},f"{outdir}/last.pt")
        with open(f"{outdir}/best.txt","w") as f: f.write(str(best))
    print("BEST",best,flush=True)

if __name__=="__main__":
    main()
