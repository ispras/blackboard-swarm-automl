import os, re, glob, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
cv2.setNumThreads(1)
ROOT=os.path.dirname(os.path.abspath(__file__))
DATA='/download_job_code/payload/sandbox/data'
OUT=os.path.join(ROOT,'cache384'); os.makedirs(OUT,exist_ok=True)
S=384
pat=re.compile(r'slice_(\d+)_(\d+)_(\d+)_')
def parse_path(p):
    d=p.split('/')
    case=d[-4]; group=d[-3]; fn=d[-1]
    m=pat.search(fn); sl=int(m.group(1)); H=int(m.group(2)); W=int(m.group(3))
    sid=f'{group}_slice_{sl:04d}'
    return case,group,sl,sid,H,W
def load_img(p):
    a=cv2.imread(p,cv2.IMREAD_UNCHANGED)
    if a is None: raise RuntimeError(p)
    a=a.astype(np.float32)
    hi=float(np.percentile(a,99.5))
    if hi<=0: hi=max(1.0,float(a.max()))
    a=np.clip(a/hi,0,1)
    a=(a*255.0).astype(np.uint8)
    a=cv2.resize(a,(S,S),interpolation=cv2.INTER_LINEAR)
    return a
def rle_decode(rle,H,W):
    flat=np.zeros(H*W,np.uint8)
    if isinstance(rle,str) and len(rle)>0:
        s=np.fromstring(rle,sep=' ').astype(np.int64)
        for st,ln in zip(s[0::2]-1,s[1::2]):
            flat[st:st+ln]=1
    return flat.reshape((H,W),order='F')
CLS=['large_bowel','small_bowel','stomach']
def build(split):
    paths=sorted(glob.glob(f'{DATA}/{split}/*/*/scans/*.png'))
    print(split,'files',len(paths))
    meta=[]; 
    for p in paths:
        case,group,sl,sid,H,W=parse_path(p); meta.append((p,case,group,sl,sid,H,W))
    # sort by group then slice
    meta.sort(key=lambda x:(x[2],x[3]))
    N=len(meta)
    imgs_path=os.path.join(OUT,f'{split}_imgs.npy')
    done_marker=os.path.join(OUT,f'.{split}_imgs_done')
    if os.path.exists(imgs_path) and os.path.exists(done_marker):
        print(split,'imgs already cached, skip',flush=True)
    else:
        imgs=np.lib.format.open_memmap(imgs_path,mode='w+',dtype=np.uint8,shape=(N,S,S))
        with Pool(4) as pool:
            for i,a in enumerate(pool.imap(load_img,[m[0] for m in meta],chunksize=16)):
                imgs[i]=a
                if i%2000==0: print(split,'img',i,flush=True)
        imgs.flush(); del imgs
        open(done_marker,'w').write('ok')
    rows=[]; groups={}
    for i,m in enumerate(meta):
        p,case,group,sl,sid,H,W=m
        rows.append(dict(row=i,id=sid,case=case,group=group,slice=sl,H=H,W=W))
        groups.setdefault(group,[]).append(i)
    df=pd.DataFrame(rows)
    prev=[-1]*N; nxt=[-1]*N
    for g,idxs in groups.items():
        for k,i in enumerate(idxs):
            prev[i]=idxs[k-1] if k>0 else i
            nxt[i]=idxs[k+1] if k<len(idxs)-1 else i
    df['prev']=prev; df['next']=nxt
    if split=='train':
        lab=pd.read_csv(f'{DATA}/train.csv')
        mp={(i,c):g for i,c,g in zip(lab['id'],lab['class'],lab['segmentation'])}
        flags=np.lib.format.open_memmap(os.path.join(OUT,'train_flags.npy'),mode='w+',dtype=np.uint8,shape=(N,S,S))
        for i,m in enumerate(meta):
            p,case,group,sl,sid,H,W=m
            fl=np.zeros((H,W),np.uint8)
            for b,c in enumerate(CLS):
                fl |= (rle_decode(mp.get((sid,c)),H,W)<<b)
            fl=cv2.resize(fl,(S,S),interpolation=cv2.INTER_NEAREST)
            flags[i]=fl
            if i%2000==0: print('train mask',i,flush=True)
        flags.flush(); del flags
        folds=pd.read_csv('/download_job_code/payload/sandbox/blackboard/betelgeuse_zeta_folds.csv')
        df=df.merge(folds,on='group',how='left')
        assert df['fold'].notna().all()
        df['fold']=df['fold'].astype(int)
    df.to_csv(os.path.join(OUT,f'{split}_index.csv'),index=False)
    print(split,'done',N)
if __name__=='__main__':
    build('test')
    build('train')
