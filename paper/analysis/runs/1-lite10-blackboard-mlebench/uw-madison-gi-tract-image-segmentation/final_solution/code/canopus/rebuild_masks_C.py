import os, sys, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
ROOT='/download_job_code/payload/sandbox'
CACHE=f'{ROOT}/data/cache_train_320'
S=320
tr=pd.read_csv(f'{ROOT}/data/train.csv')
seg={}
for c in ['large_bowel','small_bowel','stomach']:
    sub=tr[tr['class']==c]; seg[c]=dict(zip(sub.id, sub.segmentation))
index=pd.read_csv(f'{CACHE}/index.csv')
ids=index['id'].tolist(); hs=index['orig_h'].tolist(); ws=index['orig_w'].tolist()

def dec(rle,h,w):
    m=np.zeros(h*w,np.uint8)
    if isinstance(rle,str) and rle.strip() and rle==rle:
        s=rle.split(); st=np.asarray(s[0::2],np.int64)-1; ln=np.asarray(s[1::2],np.int64)
        for a,b in zip(st,ln): m[a:a+b]=1
    return m.reshape((h,w),order='C')

def one(args):
    i=args; h,w=hs[i],ws[i]; m=np.zeros((h,w),np.uint8)
    for ci,c in enumerate(['large_bowel','small_bowel','stomach']):
        rle=seg[c].get(ids[i])
        if isinstance(rle,str) and rle.strip() and rle==rle:
            m[dec(rle,h,w)>0]=ci+1
    mr=cv2.resize(m,(S,S),interpolation=cv2.INTER_NEAREST)
    return i, mr.tobytes()

masks=np.lib.format.open_memmap(f'{CACHE}/masks_u8.npy', mode='r+', dtype=np.uint8, shape=(len(ids),S,S))
with Pool(8) as p:
    for k,(i,b) in enumerate(p.imap_unordered(one, range(len(ids)), chunksize=128)):
        masks[i]=np.frombuffer(b,np.uint8).reshape(S,S)
        if k%5000==0: print(k,flush=True)
masks.flush(); print('done', len(ids))
