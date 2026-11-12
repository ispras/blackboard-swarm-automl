import argparse, json, os, sys
import numpy as np, pandas as pd, cv2
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
sys.path.insert(0,'.')
from make_submission import rle_encode
from postproc3d import clean_volume
CACHE_TE='/download_job_code/payload/sandbox/data/cache_test_320'
CLASSES=['large_bowel','small_bowel','stomach']
ap=argparse.ArgumentParser(); ap.add_argument('--probs',required=True); ap.add_argument('--config',required=True)
ap.add_argument('--out',required=True); a=ap.parse_args()
cfg=json.load(open(a.config))['per_class']
index=pd.read_csv(f'{CACHE_TE}/index.csv')
P=np.load(a.probs,mmap_mode='r')
pred=np.zeros((len(index),3,320,320),bool)
for c in range(3):
    C=cfg[str(c)]
    Q=np.asarray(P[:,c]).astype(np.float32)
    sig=C.get('sigma',0)
    if sig and sig>0:
        for cd,g in index.groupby('case_day'):
            ii=g.index.values
            Q[ii]=gaussian_filter1d(Q[ii],sigma=sig,axis=0,mode='nearest')
    pv=Q>int(C['thr']*255)
    if C.get('largest_cc'):
        for z in range(pv.shape[0]):
            if pv[z].sum()==0: continue
            l,n=ndimage.label(pv[z],structure=np.ones((3,3),np.uint8))
            if n>1:
                sizes=ndimage.sum(pv[z],l,range(1,n+1)); pv[z]=l==(1+int(np.argmax(sizes)))
    pred[:,c]=pv
for cd,g in index.groupby('case_day'):
    ii=g.index.values
    for c in range(3):
        C=cfg[str(c)]
        if C.get('min_vol') or C.get('msa'):
            pred[ii,c]=clean_volume(pred[ii,c],C.get('min_vol',0),0,C.get('msa',0))
rows=[]
for i in range(len(index)):
    r=index.iloc[i]; oh,ow=int(r.orig_h),int(r.orig_w)
    for c,cls in enumerate(CLASSES):
        m=pred[i,c]
        if m.sum()==0: rows.append((r.id,cls,'')); continue
        mr=cv2.resize(m.astype(np.uint8),(ow,oh),interpolation=cv2.INTER_NEAREST)
        rows.append((r.id,cls,rle_encode(mr,'C')))
    if i%2000==0: print(i,flush=True)
sub=pd.DataFrame(rows,columns=['id','class','predicted'])
te=pd.read_csv('/download_job_code/payload/sandbox/data/test.csv')
sub=te[['id','class']].merge(sub,on=['id','class'],how='left'); sub['predicted']=sub['predicted'].fillna('')
sub.to_csv(a.out,index=False)
print('WROTE',a.out,'nonempty',int((sub.predicted!='').sum()))
