import sys, json, argparse
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0, '.')
from metric import hd_score, points_of, CLASSES
CACHE='/download_job_code/payload/sandbox/data/cache_train_320'
ap=argparse.ArgumentParser(); ap.add_argument('--probs',required=True); ap.add_argument('--ids',required=True)
ap.add_argument('--thrs',default='0.50,0.45,0.40'); ap.add_argument('--sigmas',default='0,0.75,1.0,1.25,1.75,2.0')
a=ap.parse_args()
index=pd.read_csv(f'{CACHE}/index.csv'); mp={v:i for i,v in enumerate(index['id'])}
ids=pd.read_csv(a.ids)['id'].values; rows=np.array([mp[v] for v in ids])
vi=index.iloc[rows].reset_index(drop=True)
gt=np.asarray(np.load(f'{CACHE}/masks_u8.npy',mmap_mode='r')[rows])
P=np.asarray(np.load(a.probs,mmap_mode='r'))
thrs=[float(x) for x in a.thrs.split(',')]; sigmas=[float(x) for x in a.sigmas.split(',')]
groups=[(cd,g.index.values) for cd,g in vi.groupby('case_day')]
for sig in sigmas:
    Q=P.astype(np.float32)
    if sig>0:
        for _,ii in groups:
            for c in range(3):
                Q[ii,c]=gaussian_filter1d(Q[ii,c],sigma=sig,axis=0,mode='nearest')
    dices=[];hds=[]
    for c in range(3):
        pv=Q[:,c]>int(thrs[c]*255); g=(gt==c+1)
        inter=np.logical_and(pv,g).reshape(len(pv),-1).sum(1); s=pv.reshape(len(pv),-1).sum(1)+g.reshape(len(g),-1).sum(1)
        dices.append(float(np.where(s>0,2*inter/np.maximum(s,1),1.0).mean()))
        hds.append(float(np.mean([hd_score(points_of(pv[ii]),points_of((gt[ii]==c+1))) for _,ii in groups])))
    dice=np.mean(dices); hd=np.mean(hds); combo=0.4*dice+0.6*hd
    print(f'sigma={sig} dice={dice:.5f} hd={hd:.5f} combo={combo:.5f} per_cls={dices}', flush=True)
    Q=None
