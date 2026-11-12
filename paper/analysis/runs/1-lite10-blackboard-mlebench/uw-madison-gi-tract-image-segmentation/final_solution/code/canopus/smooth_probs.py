import argparse, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
ap=argparse.ArgumentParser(); ap.add_argument('--probs',required=True); ap.add_argument('--out',required=True); ap.add_argument('--sigma',type=float,default=1.5)
ap.add_argument('--index',default='/download_job_code/payload/sandbox/data/cache_test_320/index.csv')
a=ap.parse_args()
index=pd.read_csv(a.index)
P=np.load(a.probs,mmap_mode='r')
out=np.lib.format.open_memmap(a.out,mode='w+',dtype=np.uint8,shape=P.shape)
out[:]=np.asarray(P)
for cd,g in index.groupby('case_day'):
    ii=g.index.values
    for c in range(3):
        v=out[ii,c].astype(np.float32)
        v=gaussian_filter1d(v,sigma=a.sigma,axis=0,mode='nearest')
        out[ii,c]=np.clip(v,0,255).round().astype(np.uint8)
out.flush(); print('smoothed',a.out,float(a.sigma))
