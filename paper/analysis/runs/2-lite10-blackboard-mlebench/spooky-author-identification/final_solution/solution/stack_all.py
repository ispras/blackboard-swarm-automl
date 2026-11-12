import numpy as np, pandas as pd, glob, os, json, itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.optimize import minimize

DATA='/download_job_code/payload/sandbox/data'; ROOT='/download_job_code/payload/sandbox/workspaces'
tr=pd.read_csv(f'{DATA}/train.csv'); te=pd.read_csv(f'{DATA}/test.csv')
classes=['EAP','HPL','MWS']; y=tr['author'].map({c:i for i,c in enumerate(classes)}).values
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(tr,y))
EPS=1e-15

def collect(folders=None, include_glob=None):
    names=[]; O=[]; T=[]
    if folders is None: folders=sorted(glob.glob(f'{ROOT}/*'))
    for d in folders:
        for f in sorted(glob.glob(f'{d}/oof_*.npy')):
            base=os.path.basename(f); tag=base[4:-4]
            tf=f'{d}/test_{tag}.npy'
            if not os.path.exists(tf): continue
            try:
                o=np.load(f); t=np.load(tf)
            except Exception: continue
            if o.shape!=(len(y),3) or t.shape!=(len(te),3): continue
            if not np.all(np.isfinite(o)) or not np.all(np.isfinite(t)): continue
            o=np.clip(o,EPS,1); o=o/o.sum(1,keepdims=True)
            t=np.clip(t,EPS,1); t=t/t.sum(1,keepdims=True)
            names.append(f'{os.path.basename(d)}/{tag}'); O.append(o); T.append(t)
    return names,np.stack(O),np.stack(T)

def stack_cv(Z, C, use_log=True):
    feats=np.log(Z) if use_log else Z
    X=feats.transpose(1,0,2).reshape(len(y),-1)
    oof=np.zeros((len(y),3))
    for i,j in folds:
        m=LogisticRegression(C=C,max_iter=3000,multi_class='multinomial')
        m.fit(X[i],y[i]); oof[j]=m.predict_proba(X[j])
    return log_loss(y,oof),oof

if __name__=='__main__':
    import sys
    folders=sys.argv[1:] or None
    names,O,T=collect(folders)
    print('models',len(names))
    for n,o in zip(names,O): print(f'  {n}: {log_loss(y,o):.5f}')
    best=(9,None,None)
    for C in [0.01,0.03,0.1,0.3,1.0,3.0,10.0]:
        s,_=stack_cv(O,C,True); print('log-stack C',C,'=>',round(s,5),flush=True)
        if s<best[0]: best=(s,C,'log')
    for C in [0.1,1.0,10.0]:
        s,_=stack_cv(O,C,False); print('raw-stack C',C,'=>',round(s,5),flush=True)
        if s<best[0]: best=(s,C,'raw')
    print('BEST',best)
