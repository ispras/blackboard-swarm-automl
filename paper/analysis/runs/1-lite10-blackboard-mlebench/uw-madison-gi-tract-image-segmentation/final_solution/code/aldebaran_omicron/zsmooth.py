import sys, os, argparse
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ap=argparse.ArgumentParser()
ap.add_argument("--prob",required=True); ap.add_argument("--index",required=True)
ap.add_argument("--out",required=True); ap.add_argument("--sigma",type=float,default=1.0)
ap.add_argument("--thr",default="0.5,0.5,0.5")
a=ap.parse_args()
meta=pd.read_csv(a.index).reset_index(drop=True)
probs=np.load(a.prob,mmap_mode="r")
out=np.lib.format.open_memmap(a.out,mode="w+",dtype=np.uint8,shape=probs.shape)
meta["row"]=np.arange(len(meta))
for cd,sub in meta.groupby("case_day"):
    idxs=sub.sort_values("slice_int")["row"].values
    vol=probs[idxs].astype(np.float32)/255.0
    sm=gaussian_filter1d(vol,a.sigma,axis=0,mode="nearest")
    out[idxs]=(sm*255).round().clip(0,255).astype(np.uint8)
out.flush(); print("wrote",a.out)
