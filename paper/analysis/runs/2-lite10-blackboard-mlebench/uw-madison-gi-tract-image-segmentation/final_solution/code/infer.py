import os, sys, argparse, time, json, numpy as np, pandas as pd, torch, cv2
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0,os.path.dirname(__file__))
from common import *

def add_neighbors(df):
    df=df.sort_values(["case_day","idx"]).reset_index(drop=True)
    prev=np.arange(len(df)); nxt=np.arange(len(df))
    for cd,g in df.groupby("case_day",sort=False):
        idx=g.index.values; prev[idx[1:]]=idx[:-1]; nxt[idx[:-1]]=idx[1:]
    df=df.copy(); df["prev"]=prev; df["nxt"]=nxt
    return df

def build_test_df():
    return add_neighbors(pd.read_parquet(f"{WS}/test_files.parquet"))

def build_val_df(fold):
    from train import get_folds
    full=build_slice_df().reset_index(drop=True)
    fa=get_folds(full)
    vpos=np.where(fa==fold)[0]
    sub=full.iloc[vpos].reset_index(drop=True)
    g2l={int(g):i for i,g in enumerate(vpos)}
    prev=[g2l.get(int(g),i) for i,g in enumerate(full.loc[vpos,"prev"].values)]
    nxt=[g2l.get(int(g),i) for i,g in enumerate(full.loc[vpos,"nxt"].values)]
    sub=sub.copy(); sub["prev"]=prev; sub["nxt"]=nxt
    return sub, sub[["id"]+CLASSES].reset_index(drop=True)

class InferDataset(Dataset):
    def __init__(self, df, size=320, twofive=True, mask_df=None):
        self.df=df.reset_index(drop=True); self.size=size; self.twofive=twofive; self.mask_df=mask_df
    def __len__(self): return len(self.df)
    def _load(self,row):
        return norm_img(cv2.imread(row.path, cv2.IMREAD_UNCHANGED))
    def __getitem__(self,i):
        row=self.df.iloc[i]; chans=[]
        if self.twofive:
            for j in (int(row.prev),i,int(row.nxt)):
                rr=self.df.iloc[j] if (j!=i and j>=0) else row
                im=self._load(rr)
                if im.shape[:2]!=(self.size,self.size): im=cv2.resize(im,(self.size,self.size))
                chans.append(im)
        else:
            im=self._load(row)
            if im.shape[:2]!=(self.size,self.size): im=cv2.resize(im,(self.size,self.size))
            chans=[im]
        img=np.stack(chans,-1).astype(np.float32)/255.0
        if img.shape[2]==1: img=np.repeat(img,3,axis=2)
        img=(img-IMAGENET_MEAN)/IMAGENET_STD
        img=np.transpose(img,(2,0,1)).astype(np.float32)
        out={"img":torch.from_numpy(img),"idx":i}
        if self.mask_df is not None:
            mrow=self.mask_df.iloc[i]; masks=[]
            for c in CLASSES:
                m=rle_decode(mrow.get(c,None),(int(row.h),int(row.w)))
                if m.shape[0]!=self.size or m.shape[1]!=self.size:
                    m=cv2.resize(m,(self.size,self.size),interpolation=cv2.INTER_NEAREST)
                masks.append(m)
            out["mask"]=torch.from_numpy(np.stack(masks).astype(np.float32))
        return out

def tta_predict(model,x):
    o=torch.sigmoid(model(x))
    o=o+torch.flip(torch.sigmoid(model(torch.flip(x,[3]))),[3])
    o=o+torch.flip(torch.sigmoid(model(torch.flip(x,[2]))),[2])
    o=o+torch.flip(torch.sigmoid(model(torch.flip(x,[2,3]))),[2,3])
    return o/4.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--split",default="test")
    ap.add_argument("--fold",type=int,default=0); ap.add_argument("--tta",type=int,default=1)
    ap.add_argument("--out",default=""); ap.add_argument("--batch",type=int,default=32); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--limit",type=int,default=0); ap.add_argument("--order_ids",default=""); ap.add_argument("--outsize",type=int,default=0)
    args=ap.parse_args()
    ck=torch.load(args.ckpt,map_location="cpu"); a=ck["args"]; size=a["size"]
    import segmentation_models_pytorch as smp
    model=getattr(smp,a["arch"])(encoder_name=a["encoder"],encoder_weights=None,
            in_channels=3 if a["twofive"] else 1,classes=3)
    model.load_state_dict(ck["state"]); model=model.cuda().eval()
    if args.split=="test":
        files=build_test_df(); mask_df=None
    else:
        files,mask_df=build_val_df(args.fold)
    if args.limit:
        files=files.iloc[:args.limit].reset_index(drop=True)
        files["prev"]=files["prev"].clip(0,args.limit-1); files["nxt"]=files["nxt"].clip(0,args.limit-1)
    if args.order_ids:
        target=json.load(open(args.order_ids))
        pos={str(sid):i for i,sid in enumerate(files.id.values)}
        assert all(t in pos for t in target), ("missing", [t for t in target if t not in pos][:3])
        perm=[pos[t] for t in target]
        oldp=files.prev.values; oldn=files.nxt.values
        inv={old:k for k,old in enumerate(perm)}
        newp=[inv.get(int(oldp[o]),k) for k,o in enumerate(perm)]
        newn=[inv.get(int(oldn[o]),k) for k,o in enumerate(perm)]
        files=files.iloc[perm].reset_index(drop=True); files["prev"]=newp; files["nxt"]=newn
        if mask_df is not None: mask_df=mask_df.iloc[perm].reset_index(drop=True)
    ids=files.id.values
    outsz=args.outsize or size
    ds=InferDataset(files,size=size,twofive=bool(a["twofive"]),mask_df=mask_df)
    dl=DataLoader(ds,batch_size=args.batch,num_workers=args.workers,pin_memory=True)
    N=len(ds); probs=np.zeros((N,3,outsz,outsz),dtype=np.float16); masks=None
    if mask_df is not None: masks=np.zeros((N,3,size,size),dtype=np.uint8)
    t0=time.time(); done=0
    with torch.no_grad():
        for b in dl:
            x=b["img"].cuda(non_blocking=True)
            with torch.cuda.amp.autocast():
                p=tta_predict(model,x) if args.tta else torch.sigmoid(model(x))
            if outsz!=size:
                p=torch.nn.functional.interpolate(p.float(),size=(outsz,outsz),mode="bilinear",align_corners=False)
            idxs=b["idx"].numpy(); probs[idxs]=p.float().cpu().numpy().astype(np.float16); done+=len(idxs)
            if masks is not None: masks[idxs]=b["mask"].numpy().astype(np.uint8)
            if done % (args.batch*10) < args.batch: print(f"{done}/{N} {time.time()-t0:.1f}s",flush=True)
    out=args.out or os.path.join(os.path.dirname(args.ckpt),f"{args.split}_probs.npy")
    np.save(out,probs); np.save(out.replace(".npy","_ids.npy"),np.array(ids))
    if masks is not None:
        np.save(out.replace(".npy","_gt.npy"),masks)
        files[["id","case_day","h","w","idx"]].to_csv(out.replace(".npy","_meta.csv"),index=False)
    print("saved",out,probs.shape,flush=True)
if __name__=="__main__": main()
