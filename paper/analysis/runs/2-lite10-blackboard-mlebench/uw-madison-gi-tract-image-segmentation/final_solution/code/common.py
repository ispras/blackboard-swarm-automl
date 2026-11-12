import os, re, numpy as np, pandas as pd, cv2, torch
from torch.utils.data import Dataset
import albumentations as A

CLASSES=["large_bowel","small_bowel","stomach"]
BASE="/download_job_code/payload/sandbox"
WS=f"{BASE}/workspaces/dubhe_zeta"

def rle_decode(rle, shape):
    h,w=shape
    if not isinstance(rle,str) or rle.strip()=="":
        return np.zeros((h,w),dtype=np.uint8)
    s=np.asarray(rle.split(),dtype=np.int64)
    starts=s[0::2]-1; lengths=s[1::2]
    out=np.zeros(h*w,dtype=np.uint8)
    for st,ln in zip(starts,lengths):
        if ln>0: out[st:st+ln]=1
    return out.reshape((h,w))  # ROW-MAJOR (C-order), verified empirically

def rle_encode(mask):
    # mask HxW -> column-major flat (top->bottom then left->right)
    flat=mask.reshape(-1)  # ROW-MAJOR (C-order)
    if flat.sum()==0: return ""
    d=np.diff(np.concatenate([[0],flat,[0]]))
    starts=np.where(d==1)[0]+1
    ends=np.where(d==-1)[0]
    lengths=ends-starts+1
    return " ".join(f"{s} {l}" for s,l in zip(starts,lengths))

def build_slice_df():
    files=pd.read_parquet(f"{WS}/train_files.parquet")
    tr=pd.read_parquet(f"{WS}/train_rows.parquet")
    piv=tr.pivot(index="id",columns="class",values="segmentation").reset_index()
    df=files.merge(piv,on="id",how="left")
    df=df.sort_values(["case_day","idx"]).reset_index(drop=True)
    # neighbor indices within same case_day sorted by idx
    prev=np.arange(len(df)); nxt=np.arange(len(df))
    start=0
    for cd,g in df.groupby("case_day",sort=False):
        idx=g.index.values
        prev[idx[1:]]=idx[:-1]
        nxt[idx[:-1]]=idx[1:]
    df["prev"]=prev; df["nxt"]=nxt
    df["case_code"]=df["case"].astype("category").cat.codes
    df["cd_code"]=df["case_day"].astype("category").cat.codes
    return df

def norm_img(img):
    img=img.astype(np.float32)
    p1,p99=np.percentile(img,1),np.percentile(img,99)
    if p99-p1<1e-3: p99=p1+1.0
    img=np.clip((img-p1)/(p99-p1),0,1)
    return (img*255).astype(np.uint8)

IMAGENET_MEAN=np.array([0.485,0.456,0.406],dtype=np.float32)
IMAGENET_STD=np.array([0.229,0.224,0.225],dtype=np.float32)

def get_train_aug(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=20,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
    ], additional_targets={"m0":"mask","m1":"mask","m2":"mask"})

class GIDataset(Dataset):
    def __init__(self, df, size=320, train=True, indices=None, twofive=True):
        self.df=df.reset_index(drop=True) if indices is None else df.iloc[indices].reset_index(drop=True)
        self.size=size; self.train=train; self.twofive=twofive
        self.aug=get_train_aug(size) if train else None
        if indices is not None:
            g2l={int(g):l for l,g in enumerate(indices)}
            self.df["prev"]=self.df["prev"].map(lambda g: g2l.get(int(g),-1))
            self.df["nxt"]=self.df["nxt"].map(lambda g: g2l.get(int(g),-1))
    def __len__(self): return len(self.df)
    def _load(self, row):
        img=cv2.imread(row.path, cv2.IMREAD_UNCHANGED)
        if img is None: raise RuntimeError(row.path)
        return img
    def __getitem__(self, i):
        row=self.df.iloc[i]
        imgs=[]
        if self.twofive:
            for j in (int(row.prev), i, int(row.nxt)):
                r=self.df.iloc[j] if (j!=i and j>=0) else row
                imgs.append(norm_img(self._load(r)))
        else:
            imgs=[norm_img(self._load(row))]
        # resize each and stack channels
        chans=[]
        for im in imgs:
            if im.shape[:2]!=(self.size,self.size):
                im=cv2.resize(im,(self.size,self.size),interpolation=cv2.INTER_LINEAR)
            chans.append(im)
        img=np.stack(chans,axis=-1)  # H,W,C
        masks=[]
        for c in CLASSES:
            rle=row.get(c,None)
            m=rle_decode(rle,(int(row.h),int(row.w)))
            if m.shape[0]!=self.size or m.shape[1]!=self.size:
                m=cv2.resize(m,(self.size,self.size),interpolation=cv2.INTER_NEAREST)
            masks.append(m)
        m0,m1,m2=masks
        if self.train and self.aug is not None:
            r=self.aug(image=img,m0=m0,m1=m1,m2=m2)
            img=r["image"]; m0,m1,m2=r["m0"],r["m1"],r["m2"]
        img=img.astype(np.float32)/255.0
        if img.shape[2]==1:
            img=np.repeat(img,3,axis=2)
        img=(img-IMAGENET_MEAN)/IMAGENET_STD
        img=np.transpose(img,(2,0,1)).astype(np.float32)
        mask=np.stack([m0,m1,m2],axis=0).astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(mask)

def dice_np(pred, gt, eps=1e-6):
    # pred,gt bool 2d
    inter=(pred&gt).sum()
    return (2*inter+eps)/(pred.sum()+gt.sum()+eps)
