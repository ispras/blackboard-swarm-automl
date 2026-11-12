import os, sys, cv2, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, timm
from sklearn.metrics import roc_auc_score, matthews_corrcoef
sys.path.insert(0,f"/download_job_code/payload/sandbox/workspaces/castor_alpha")
from build_crops import crop_square, build_index, find_box, OFF, FPS, OUT
R="/download_job_code/payload/sandbox"; W=f"{R}/workspaces/castor_alpha"
ARCH=os.environ.get("ARCH","resnet18")
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
GSCALE=float(os.environ.get("GSCALE","3.6")); GSHIFT=float(os.environ.get("GSHIFT","0.9"))

class Net(nn.Module):
    def __init__(self, ntab, arch="resnet18"):
        super().__init__(); self.enc=timm.create_model(arch,pretrained=False,num_classes=0); d=self.enc.num_features
        self.head=nn.Sequential(nn.Linear(d*3+ntab,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,1))
    def forward(self,a,b,t):
        ea=self.enc(a); eb=self.enc(b); return self.head(torch.cat([ea,eb,torch.abs(ea-eb),t],1)).squeeze(1)

def norm_batch(arr):
    a=arr[...,::-1].astype(np.float32)/255.0; a=(a-MEAN)/STD; return torch.from_numpy(a.transpose(0,3,1,2))

def load_tab(df, kind, split, stats):
    f=pd.read_parquet(f"{W}/{kind}_{split}_feats.parquet")
    cols=stats["cols"]; f=f[["contact_id"]+cols]
    m=df[["contact_id"]].merge(f,on="contact_id",how="left")
    X=m[cols].to_numpy(np.float32); X=np.where(np.isfinite(X),X,stats["med"]); X=(X-stats["mu"])/stats["sd"]
    return X.astype(np.float32)

def make_crops(frame,idx,view,fr,row,kind):
    b1=find_box(idx,view,fr,int(row.pid))
    if kind.startswith("ground"):
        if b1 is None: return np.zeros((OUT,OUT,3),np.uint8)
        l,t,w,hh=b1; cx=l+w/2; cy=t+hh/2+GSHIFT*hh; side=max(GSCALE*max(w,hh),40.0)
        return crop_square(frame,cx,cy,side)
    b2=find_box(idx,view,fr,int(row.pid2))
    if b1 is None and b2 is None: return np.zeros((OUT,OUT,3),np.uint8)
    if b1 is not None and b2 is not None:
        l=min(b1[0],b2[0]); t=min(b1[1],b2[1]); r=max(b1[0]+b1[2],b2[0]+b2[2]); btm=max(b1[1]+b1[3],b2[1]+b2[3])
        side=max((r-l)*1.8,(btm-t)*1.8,60.0); return crop_square(frame,(l+r)/2,(t+btm)/2,side)
    bb=b1 if b1 is not None else b2; l,t,w,hh=bb; side=max(4.0*max(w,hh),60.0)
    return crop_square(frame,l+w/2,t+hh/2+0.5*hh,side)

def run(kind,split,outname,bs=256):
    stats=np.load(f"{W}/tabstats_{kind}.npy",allow_pickle=True).item()
    if split=="train":
        lab=pd.read_csv(f"{R}/data/train_labels.csv")
        if kind.startswith("ground"):
            df=lab[lab.nfl_player_id_2.astype(str)=="G"].copy(); df["pid"]=df.nfl_player_id_1.astype(int); df["pid2"]=-1
        else:
            df=lab[lab.nfl_player_id_2.astype(str)!="G"].copy(); df["pid"]=df.nfl_player_id_1.astype(int); df["pid2"]=df.nfl_player_id_2.astype(int)
        df["y"]=df.contact.astype(np.int8)
        folds=pd.read_csv(f"{W}/folds.csv"); df=df.merge(folds,on="game_play",how="left")
    else:
        ss=pd.read_csv(f"{R}/data/sample_submission.csv"); parts=ss.contact_id.str.split("_",n=4,expand=True)
        ss["game_play"]=parts[0]+"_"+parts[1]; ss["step"]=parts[2].astype(int); ss["pid"]=parts[3].astype(int)
        if kind.startswith("ground"): df=ss[parts[4]=="G"].copy(); df["pid2"]=-1
        else: df=ss[parts[4]!="G"].copy(); df["pid2"]=parts[4].loc[df.index].astype(int)
        df["y"]=0; df["fold"]=-1
    df=df.reset_index(drop=True)
    X=load_tab(df,kind,split,stats)
    print("split",split,"rows",len(df),"ntab",X.shape[1],flush=True)
    models={}
    for f in df.fold.unique():
        if split=="train":
            ck=torch.load(f"{W}/models/{kind}_{ARCH}_f{f}.pt",map_location="cuda")
            m=Net(ck["ntab"],ck.get("arch",ARCH)).cuda(); m.load_state_dict(ck["state"]); m.eval(); models[f]=m
        else:
            models[f]=None
    if split=="test":
        cks=[torch.load(f"{W}/models/{kind}_{ARCH}_f{f}.pt",map_location="cuda") for f in range(5)]
        ms=[]
        for ck in cks:
            m=Net(ck["ntab"],ck.get("arch",ARCH)).cuda(); m.load_state_dict(ck["state"]); m.eval(); ms.append(m)
    pred=np.full(len(df),np.nan,np.float32)
    for gp,gdf in df.groupby("game_play",sort=False):
        h=pickle.load(open(f"{W}/helm_cache/{split}/{gp}.pkl","rb")); idx=build_index(h)
        if split=="train":
            m=models[int(gdf.fold.iloc[0])]
        buf=[]; idxs=[]
        def flush():
            if not buf: return
            A=norm_batch(np.stack([b[0] for b in buf])); B=norm_batch(np.stack([b[1] for b in buf]))
            T=torch.from_numpy(np.stack([b[2] for b in buf])); 
            with torch.no_grad():
                if split=="train": p=torch.sigmoid(m(A.cuda(),B.cuda(),T.cuda())).cpu().numpy()
                else: p=np.mean([torch.sigmoid(mm(A.cuda(),B.cuda(),T.cuda())).cpu().numpy() for mm in ms],0)
            for j,ii in enumerate(idxs): pred[ii]=p[j]
            buf.clear(); idxs.clear()
        for step,gstep in gdf.groupby("step",sort=True):
            fr=int(round(OFF+float(step)*FPS))
            frames={}
            for view in ["Sideline","Endzone"]:
                frames[view]=cv2.imread(f"{W}/frames/{split}/{gp}_{view}_{int(step):04d}.jpg")
                if frames[view] is None: frames[view]=np.zeros((720,1280,3),np.uint8)
            for ii,row in gstep.iterrows():
                a=make_crops(frames["Sideline"],idx,"Sideline",fr,row,kind)
                b=make_crops(frames["Endzone"],idx,"Endzone",fr,row,kind)
                buf.append((a,b,X[ii])); idxs.append(ii)
                if len(buf)>=bs: flush()
        flush()
        print("done",gp,flush=True)
    out=df[["contact_id","game_play","step","pid","pid2","y","fold"]].copy(); out["pred"]=pred
    out.to_parquet(f"{W}/{outname}.parquet",index=False)
    if split=="train":
        print("OOF AUC",roc_auc_score(out.y,out.pred),flush=True)
        best=(0,0.5)
        for th in np.unique(np.quantile(out.pred,np.linspace(0.5,0.9995,500))):
            mm=matthews_corrcoef(out.y,(out.pred>=th).astype(int))
            if mm>best[0]: best=(mm,th)
        print("OOF MCC",best,flush=True)
    print("SAVED",outname,flush=True)

if __name__=="__main__":
    run(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]) if len(sys.argv)>4 else 256)
