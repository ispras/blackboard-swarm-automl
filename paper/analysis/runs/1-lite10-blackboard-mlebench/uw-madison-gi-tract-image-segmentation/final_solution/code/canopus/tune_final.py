import sys, json, argparse
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
sys.path.insert(0,'.')
from metric import hd_score, points_of, CLASSES
CACHE='/download_job_code/payload/sandbox/data/cache_train_320'
ap=argparse.ArgumentParser(); ap.add_argument('--probs',required=True); ap.add_argument('--ids',required=True)
ap.add_argument('--out',required=True); a=ap.parse_args()
index=pd.read_csv(f'{CACHE}/index.csv'); mp={v:i for i,v in enumerate(index['id'])}
ids=pd.read_csv(a.ids)['id'].values; rows=np.array([mp[v] for v in ids])
vi=index.iloc[rows].reset_index(drop=True)
gt=np.asarray(np.load(f'{CACHE}/masks_u8.npy',mmap_mode='r')[rows])
P=np.asarray(np.load(a.probs,mmap_mode='r')).astype(np.float32)
groups=[(cd,g.index.values) for cd,g in vi.groupby('case_day')]
sigmas=[0.0,0.75,1.25,1.75,2.5]; thrs=[0.30,0.35,0.40,0.45,0.50,0.55,0.60]
def largest_cc_slice(m):
    out=np.zeros_like(m)
    for z in range(m.shape[0]):
        if m[z].sum()==0: continue
        l,n=ndimage.label(m[z],structure=np.ones((3,3),np.uint8))
        if n<=1: out[z]=m[z]; continue
        sizes=ndimage.sum(m[z],l,range(1,n+1)); out[z]=l==(1+int(np.argmax(sizes)))
    return out
cfg={}
for c in range(3):
    g=(gt==c+1); gpts=[points_of(g[ii]) for _,ii in groups]
    best=None
    for sig in sigmas:
        Q=P[:,c].copy()
        if sig>0:
            for _,ii in groups: Q[ii]=gaussian_filter1d(Q[ii],sigma=sig,axis=0,mode='nearest')
        for t in thrs:
            pv=Q>int(t*255)
            inter=np.logical_and(pv,g).reshape(len(pv),-1).sum(1); s=pv.reshape(len(pv),-1).sum(1)+g.reshape(len(g),-1).sum(1)
            dice=float(np.where(s>0,2*inter/np.maximum(s,1),1.0).mean())
            hd=float(np.mean([hd_score(points_of(pv[ii]),gpts[k]) for k,(_,ii) in enumerate(groups)]))
            combo=0.4*dice+0.6*hd
            if best is None or combo>best['combo']:
                best=dict(sigma=sig,thr=t,dice=dice,hd=hd,combo=combo)
        # stomach-only largest-cc experiment
        if c==2:
            for t in thrs:
                pv=Q>int(t*255); pv2=largest_cc_slice(pv)
                inter=np.logical_and(pv2,g).reshape(len(pv2),-1).sum(1); s=pv2.reshape(len(pv2),-1).sum(1)+g.reshape(len(g),-1).sum(1)
                dice=float(np.where(s>0,2*inter/np.maximum(s,1),1.0).mean())
                hd=float(np.mean([hd_score(points_of(pv2[ii]),gpts[k]) for k,(_,ii) in enumerate(groups)]))
                combo=0.4*dice+0.6*hd
                if combo>best['combo']:
                    best=dict(sigma=sig,thr=t,dice=dice,hd=hd,combo=combo,largest_cc=True)
    best.setdefault('largest_cc',False)
    cfg[c]=best; print('class',CLASSES[c],json.dumps(best),flush=True)
overall=dict(dice=float(np.mean([cfg[c]['dice'] for c in range(3)])),hd=float(np.mean([cfg[c]['hd'] for c in range(3)])),combo=float(np.mean([cfg[c]['combo'] for c in range(3)])))
print('OVERALL',json.dumps(overall),flush=True)
json.dump(dict(per_class=cfg,overall=overall),open(a.out,'w'),indent=1)
