import os, sys, argparse, numpy as np, pandas as pd, cv2
sys.path.insert(0,os.path.dirname(__file__))
from common import *

def postproc_mask(m, min_area=0, fill=True):
    if m.sum()==0: return m
    if fill:
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    if min_area>0:
        n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
        out=np.zeros_like(m)
        for k in range(1,n):
            if stats[k,cv2.CC_STAT_AREA]>=min_area: out[lab==k]=1
        m=out
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--probs",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--thr",default="0.5,0.5,0.5")
    ap.add_argument("--min_area",type=int,default=20)
    ap.add_argument("--split",default="test")
    args=ap.parse_args()
    thr=np.array([float(x) for x in args.thr.split(",")],dtype=np.float32)
    probs=np.load(args.probs, mmap_mode="r")
    ids=np.load(args.probs.replace(".npy","_ids.npy"),allow_pickle=True)
    if args.split=="test":
        files=pd.read_parquet(f"{WS}/test_files.parquet").set_index("id")
    else:
        files=pd.read_parquet(f"{WS}/train_files.parquet").set_index("id")
    size=probs.shape[-1]
    rows=[]
    for i,sid in enumerate(ids):
        sid=str(sid)
        h=int(files.loc[sid,"h"]); w=int(files.loc[sid,"w"])
        for c,cls in enumerate(CLASSES):
            m=(np.asarray(probs[i,c])>thr[c]).astype(np.uint8)
            m=postproc_mask(m,args.min_area)
            if m.shape[0]!=h or m.shape[1]!=w:
                m=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
            rows.append((sid,cls,rle_encode(m)))
    out=pd.DataFrame(rows,columns=["id","class","predicted"])
    if args.split=="test":
        order=pd.read_csv(f"{BASE}/data/test.csv")
        out=order.merge(out,on=["id","class"],how="left")
        out["predicted"]=out["predicted"].fillna("")
    out.to_csv(args.out,index=False)
    print("wrote",args.out,len(out),"nonempty", (out.predicted.fillna("")!="").sum(),flush=True)

if __name__=="__main__":
    main()
