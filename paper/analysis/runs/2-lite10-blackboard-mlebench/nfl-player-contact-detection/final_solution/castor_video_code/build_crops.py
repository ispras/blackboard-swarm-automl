import os, sys, cv2, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
R="/download_job_code/payload/sandbox"; W=f"{R}/workspaces/castor_alpha"
OFF=290.0; FPS=5.994; OUT=128
GSCALE=float(os.environ.get("GSCALE","3.6")); GSHIFT=float(os.environ.get("GSHIFT","0.9"))

def get_samples(kind, split, neg_mult=4.0, seed=42):
    if split=="train":
        lab=pd.read_csv(f"{R}/data/train_labels.csv")
        if kind.startswith("ground"):
            df=lab[lab.nfl_player_id_2.astype(str)=="G"].copy(); df["pid"]=df.nfl_player_id_1.astype(int); df["pid2"]=-1
        else:
            df=lab[lab.nfl_player_id_2.astype(str)!="G"].copy(); df["pid"]=df.nfl_player_id_1.astype(int); df["pid2"]=df.nfl_player_id_2.astype(int)
        df["y"]=df.contact.astype(np.int8)
        rng=np.random.RandomState(seed)
        pos=df[df.y==1]; neg=df[df.y==0]
        n_neg=int(min(len(neg), neg_mult*len(pos)))
        if kind=="pair":
            trk=pd.read_parquet(f"{R}/workspaces/achernar_sigma/train_tracking_features.parquet",columns=["contact_id","dist"])
            neg=neg.merge(trk,on="contact_id",how="left")
            n_hard=int(n_neg*0.6)
            hard=neg[neg["dist"]<8.0]
            if len(hard)>n_hard: hard=hard.iloc[rng.choice(len(hard),size=n_hard,replace=False)]
            rest=neg.drop(hard.index)
            n_rand=n_neg-len(hard)
            rand=rest.iloc[rng.choice(len(rest),size=min(n_rand,len(rest)),replace=False)]
            neg=pd.concat([hard,rand],ignore_index=True).drop(columns=["dist"])
        else:
            neg=neg.iloc[rng.choice(len(neg),size=n_neg,replace=False)]
        s=pd.concat([pos,neg],ignore_index=True)
        folds=pd.read_csv(f"{W}/folds.csv"); s=s.merge(folds,on="game_play",how="left")
    else:
        ss=pd.read_csv(f"{R}/data/sample_submission.csv")
        parts=ss.contact_id.str.split("_",n=4,expand=True)
        ss["game_play"]=parts[0]+"_"+parts[1]; ss["step"]=parts[2].astype(int)
        ss["pid"]=parts[3].astype(int)
        if kind.startswith("ground"):
            s=ss[parts[4]=="G"].copy(); s["pid2"]=-1
        else:
            s=ss[parts[4]!="G"].copy(); s["pid2"]=parts[4].loc[s.index].astype(int)
        s["y"]=0; s["fold"]=-1
    s=s.reset_index(drop=True); s["sample_id"]=np.arange(len(s))
    return s[["sample_id","contact_id","game_play","step","pid","pid2","y","fold"]]

def crop_square(frame,cx,cy,side,out=OUT):
    H,Wd=frame.shape[:2]; half=side/2.0
    x0=int(round(cx-half)); y0=int(round(cy-half)); x1=int(round(cx+half)); y1=int(round(cy+half))
    pl=max(0,-x0); pt=max(0,-y0); pr=max(0,x1-Wd); pb=max(0,y1-H)
    x0=max(0,x0); y0=max(0,y0); x1=min(Wd,x1); y1=min(H,y1)
    if x1<=x0 or y1<=y0: return np.zeros((out,out,3),np.uint8)
    c=frame[y0:y1,x0:x1]
    if pl or pt or pr or pb: c=cv2.copyMakeBorder(c,pt,pb,pl,pr,cv2.BORDER_CONSTANT,value=(0,0,0))
    return cv2.resize(c,(out,out),interpolation=cv2.INTER_AREA)

def build_index(h):
    idx={}
    for r in h.itertuples(index=False):
        idx.setdefault((r.view,int(r.frame)),{})[int(r.nfl_player_id)]=(float(r.left),float(r.top),float(r.width),float(r.height))
    return idx

def find_box(idx,view,frame,pid,maxoff=4):
    for off in [0,-1,1,-2,2,-3,3,-4,4]:
        d=idx.get((view,frame+off))
        if d is not None and pid in d: return d[pid]
    return None

def process(task):
    gp,df,split,kind,mmpath,shape=task
    import numpy as np
    mm=np.lib.format.open_memmap(mmpath,mode="r+")
    h=pickle.load(open(f"{W}/helm_cache/{split}/{gp}.pkl","rb")); idx=build_index(h)
    views=["Sideline","Endzone"]
    for step,gstep in df.groupby("step",sort=True):
        fr=int(round(OFF+float(step)*FPS))
        for vi,view in enumerate(views):
            fp=f"{W}/frames/{split}/{gp}_{view}_{int(step):04d}.jpg"
            frame=cv2.imread(fp)
            if frame is None: frame=np.zeros((720,1280,3),np.uint8)
            for row in gstep.itertuples(index=False):
                sid=int(row.sample_id)
                b1=find_box(idx,view,fr,int(row.pid))
                if kind.startswith("ground"):
                    if b1 is None:
                        mm[sid,vi]=0; continue
                    l,t,w,hh=b1; cx=l+w/2; cy=t+hh/2+GSHIFT*hh
                    side=max(GSCALE*max(w,hh),40.0)
                    mm[sid,vi]=crop_square(frame,cx,cy,side)
                else:
                    p2=int(row.pid2); b2=find_box(idx,view,fr,p2)
                    if b1 is None and b2 is None:
                        mm[sid,vi]=0; continue
                    if b1 is not None and b2 is not None:
                        l=min(b1[0],b2[0]); t=min(b1[1],b2[1]); r=max(b1[0]+b1[2],b2[0]+b2[2]); btm=max(b1[1]+b1[3],b2[1]+b2[3])
                        side=max((r-l)*1.8,(btm-t)*1.8,60.0); cx=(l+r)/2; cy=(t+btm)/2
                        mm[sid,vi]=crop_square(frame,cx,cy,side)
                    else:
                        bb=b1 if b1 is not None else b2
                        l,t,w,hh=bb; cx=l+w/2; cy=t+hh/2+0.5*hh; side=max(4.0*max(w,hh),60.0)
                        mm[sid,vi]=crop_square(frame,cx,cy,side)
    mm.flush(); del mm
    return gp,len(df)

if __name__=="__main__":
    kind=sys.argv[1]; split=sys.argv[2]
    neg_mult=float(sys.argv[3]) if len(sys.argv)>3 else 4.0
    s=get_samples(kind,split,neg_mult)
    n=len(s)
    path=f"{W}/crops/{kind}_{split}.npy"
    shape=(n,2,OUT,OUT,3)
    mm=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint8,shape=shape); del mm
    s.to_parquet(f"{W}/crops/{kind}_{split}_meta.parquet",index=False)
    tasks=[(gp,df,split,kind,path,shape) for gp,df in s.groupby("game_play",sort=False)]
    print("kind",kind,"split",split,"N",n,"plays",len(tasks),flush=True)
    with Pool(6) as p:
        for i,r in enumerate(p.imap_unordered(process,tasks,chunksize=1)):
            print(i,r,flush=True)
    print("DONE",flush=True)
