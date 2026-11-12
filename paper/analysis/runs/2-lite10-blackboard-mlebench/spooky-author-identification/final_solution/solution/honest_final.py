import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from stack_all import collect, y, te
EPS=1e-15; OUT='/download_job_code/payload/sandbox/workspaces/canopus_kappa'; FS='/download_job_code/payload/sandbox/final_solution'
def norm(p): p=np.clip(p,EPS,1); return p/p.sum(1,keepdims=True)
DER=('partial','stack','blend','candidate','nested','meta','final','opt')
names_all,O_all,T_all=collect()
keep=[]
for k,n in enumerate(names_all):
    t=n.split('/')[-1].lower()
    if any(x in t for x in DER) or not np.isfinite(O_all[k]).all(): continue
    if log_loss(y,norm(O_all[k]))>0.75: continue
    keep.append(k)
ind={k:log_loss(y,norm(O_all[k])) for k in keep}; sel=[]
for k in sorted(keep,key=lambda k:ind[k]):
    if any(np.corrcoef(O_all[k].ravel(),O_all[j].ravel())[0,1]>0.998 for j in sel): continue
    sel.append(k)
sel=sorted(sel)
names=[names_all[k] for k in sel]; O=np.stack([norm(O_all[k]) for k in sel]); T=np.stack([norm(T_all[k]) for k in sel])
print('honest base n',len(names),flush=True)
for n in names: print('  ',n,round(log_loss(y,norm(O_all[names_all.index(n)])),5),flush=True)
def feat(P):
    L=np.log(P); return np.concatenate([L,P],2).transpose(1,0,2).reshape(len(P[0]),-1)
X=feat(O); XT=feat(T)
seeds=[42,0,1,7]
Cs=[0.1,0.2,0.3,0.5,1.0]
avg={C:[] for C in Cs}
for seed in seeds:
    skf=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y))
    for C in Cs:
        o=np.zeros((len(y),3))
        for i,j in skf:
            m=LogisticRegression(C=C,max_iter=8000).fit(X[i],y[i]); o[j]=m.predict_proba(X[j])
        avg[C].append(log_loss(y,o))
    print('seed',seed,{C:round(avg[C][-1],5) for C in Cs},flush=True)
means={C:float(np.mean(v)) for C,v in avg.items()}; print('mean',{k:round(v,5) for k,v in means.items()},flush=True)
C=min(means,key=means.get); print('chosen C',C,flush=True)
# XGB oof (shared folds seed42)
cpath=f'{OUT}/xgb_honest.npy'
if os.path.exists(cpath): xg=np.load(cpath)
else:
    xg=np.zeros((len(y),3)); folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
    for i,j in folds:
        g=XGBClassifier(n_estimators=200,learning_rate=0.03,max_depth=3,subsample=0.8,colsample_bytree=0.6,reg_lambda=1.0,objective='multi:softprob',tree_method='hist',n_jobs=2,random_state=42,verbosity=0)
        g.fit(X[i],y[i]); xg[j]=g.predict_proba(X[j])
    np.save(cpath,xg)
print('XGB',log_loss(y,norm(xg)),flush=True)
for w in [0.8,0.85,0.9,0.95,1.0]:
    vals=[]
    for seed in seeds:
        skf=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)); o=np.zeros((len(y),3))
        for i,j in skf:
            m=LogisticRegression(C=C,max_iter=8000).fit(X[i],y[i]); o[j]=norm(w*m.predict_proba(X[j])+(1-w)*xg[j])
        vals.append(log_loss(y,o))
    print('blend w',w,'mean',round(float(np.mean(vals)),5),[round(v,5) for v in vals],flush=True)
