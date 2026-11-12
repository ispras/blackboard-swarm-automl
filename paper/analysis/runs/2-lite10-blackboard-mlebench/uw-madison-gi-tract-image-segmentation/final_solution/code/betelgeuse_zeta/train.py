import os, sys, time, math, argparse, random
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
cv2.setNumThreads(1)

ROOT=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(ROOT,'cache384')
CLS=['large_bowel','small_bowel','stomach']

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class GIDataset(Dataset):
    def __init__(self, index, imgs_path, flags_path=None, train=False):
        self.index=index.reset_index(drop=True)
        self.imgs=np.load(imgs_path,mmap_mode='r')
        self.flags=np.load(flags_path,mmap_mode='r') if flags_path else None
        self.train=train
        self.aug=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.12,rotate_limit=15,border_mode=0,value=0,mask_value=0,p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.15,contrast_limit=0.15,p=0.3),
            A.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,mask_fill_value=0,p=0.2),
        ])
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        r=self.index.iloc[i]
        a=np.stack([self.imgs[int(r.prev)],self.imgs[int(r.row)],self.imgs[int(r.next)]],axis=-1)
        if self.flags is not None:
            fl=np.asarray(self.flags[int(r.row)]).astype(np.uint8)
            m=np.stack([(fl>>b)&1 for b in range(3)],axis=-1).astype(np.float32)
        else:
            m=np.zeros((a.shape[0],a.shape[1],3),np.float32)
        if self.train:
            out=self.aug(image=a,mask=m); a=out['image']; m=out['mask']
            m=(m>0.5).astype(np.float32)
        a=a.astype(np.float32)/255.0
        a=(a-0.5)/0.5
        return torch.from_numpy(a.transpose(2,0,1)), torch.from_numpy(m.transpose(2,0,1))

def dice_np(pred,gt):
    inter=(pred&gt).sum(); s=pred.sum()+gt.sum()
    if s==0: return 1.0
    return 2.0*inter/s

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); dsum={c:0.0 for c in CLS}; n={c:0 for c in CLS}; dpos={c:0.0 for c in CLS}; npos={c:0 for c in CLS}
    for x,y in loader:
        x=x.to(dev,non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16):
            lo=model(x)
        p=(torch.sigmoid(lo.float())>0.5).cpu().numpy().astype(np.uint8)
        y=y.numpy().astype(np.uint8)
        for b,c in enumerate(CLS):
            for j in range(p.shape[0]):
                pp=p[j,b]; gg=y[j,b]
                d=dice_np(pp,gg); dsum[c]+=d; n[c]+=1
                if gg.sum()>0: dpos[c]+=d; npos[c]+=1
    model.train()
    dall=np.mean([dsum[c]/max(1,n[c]) for c in CLS])
    dposm=np.mean([dpos[c]/max(1,npos[c]) for c in CLS])
    return dall,dposm,{c:round(dsum[c]/max(1,n[c]),4) for c in CLS}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--bs',type=int,default=32)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--encoder',default='efficientnet-b3')
    ap.add_argument('--arch',default='UnetPlusPlus')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--out',default=None)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--val_limit',type=int,default=0)
    ap.add_argument('--bench',type=int,default=0)
    ap.add_argument('--init',default='')
    ap.add_argument('--log_every',type=int,default=200)
    ap.add_argument('--warm_epochs',type=float,default=1.0)
    args=ap.parse_args()
    set_seed(42+args.fold)
    dev='cuda'
    idx=pd.read_csv(os.path.join(CACHE,'train_index.csv'))
    if args.limit>0: idx=idx.head(args.limit)
    if args.fold<0:
        tr=idx.reset_index(drop=True); va=idx.sample(n=min(1024,len(idx)),random_state=0).reset_index(drop=True)
    else:
        tr=idx[idx.fold!=args.fold].reset_index(drop=True)
        va=idx[idx.fold==args.fold].reset_index(drop=True)
    if args.val_limit>0: va=va.head(args.val_limit)
    print('train',len(tr),'val',len(va),flush=True)
    dtr=GIDataset(tr,os.path.join(CACHE,'train_imgs.npy'),os.path.join(CACHE,'train_flags.npy'),train=True)
    dva=GIDataset(va,os.path.join(CACHE,'train_imgs.npy'),os.path.join(CACHE,'train_flags.npy'),train=False)
    ltr=DataLoader(dtr,batch_size=args.bs,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True,persistent_workers=args.workers>0,prefetch_factor=4 if args.workers>0 else None)
    lva=DataLoader(dva,batch_size=args.bs*2,shuffle=False,num_workers=max(2,args.workers//2),pin_memory=True)
    Model=getattr(smp,args.arch)
    model=Model(encoder_name=args.encoder,encoder_weights='imagenet',in_channels=3,classes=3).to(dev)
    if args.init:
        st=torch.load(args.init,map_location='cpu'); model.load_state_dict(st['model']); print('init from',args.init,flush=True)
    bce=nn.BCEWithLogitsLoss()
    dice=smp.losses.DiceLoss(mode='multilabel')
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    steps=len(ltr); total=steps*args.epochs; warm=max(1,int(steps*args.warm_epochs))
    def lr_at(it):
        if it<warm: return args.lr*it/max(1,warm)
        p=(it-warm)/max(1,total-warm); return args.lr*(0.5*(1+math.cos(math.pi*p)))
    scaler=torch.cuda.amp.GradScaler()
    out=args.out or os.path.join(ROOT,f'runs/{args.arch}_{args.encoder}_f{args.fold}')
    os.makedirs(out,exist_ok=True)
    best=-1; it=0; t0=time.time()
    for ep in range(args.epochs):
        model.train(); run=0.0; cnt=0
        for x,y in ltr:
            lr=lr_at(it)
            for g in opt.param_groups: g['lr']=lr
            x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                lo=model(x); loss=bce(lo,y)+dice(lo,y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt); scaler.update()
            run+=loss.item(); cnt+=1; it+=1
            if args.log_every and cnt%args.log_every==0: print(f'  step {cnt}/{steps} loss {run/cnt:.4f} lr {lr:.2e} t {time.time()-t0:.0f}s',flush=True)
            if args.bench and cnt>=args.bench: break
        dall,dpos,per=evaluate(model,lva,dev)
        print(f'ep {ep} loss {run/max(1,cnt):.4f} dice_all {dall:.4f} dice_pos {dpos:.4f} {per} lr {lr:.2e} t {time.time()-t0:.0f}s',flush=True)
        score=dall
        if score>best:
            best=score
            torch.save({'model':model.state_dict(),'args':vars(args),'best':best,'ep':ep},os.path.join(out,'best.pth'))
        torch.save({'model':model.state_dict(),'args':vars(args),'best':best,'ep':ep},os.path.join(out,'last.pth'))
        if args.bench: break
    print('BEST',best,flush=True)

if __name__=='__main__': main()
