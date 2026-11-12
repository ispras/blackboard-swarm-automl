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
names_all,O_all,T_all=collect(); keep=[]
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
def feat(P):
    L=np.log(P); return np.concatenate([L,P],2).transpose(1,0,2).reshape(len(P[0]),-1)
X=feat(O); XT=feat(T)
C=0.1; W=0.8
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y))
lr_oof=np.zeros((len(y),3)); xg_oof=np.zeros((len(y),3))
for i,j in folds:
    m=LogisticRegression(C=C,max_iter=8000).fit(X[i],y[i]); lr_oof[j]=m.predict_proba(X[j])
    g=XGBClassifier(n_estimators=200,learning_rate=0.03,max_depth=3,subsample=0.8,colsample_bytree=0.6,reg_lambda=1.0,objective='multi:softprob',tree_method='hist',n_jobs=2,random_state=42,verbosity=0)
    g.fit(X[i],y[i]); xg_oof[j]=g.predict_proba(X[j])
oof=norm(W*lr_oof+(1-W)*xg_oof)
cv=log_loss(y,oof); cv_lr=log_loss(y,norm(lr_oof)); cv_xg=log_loss(y,norm(xg_oof))
print('n',len(names),'LR',cv_lr,'XGB',cv_xg,'BLEND',cv,flush=True)
# repeated-seed robustness of the exact blend recipe
vals=[]
for seed in [0,1,7]:
    skf=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y)); o=np.zeros((len(y),3))
    # for other seeds reuse fixed base OOF (each row still OOF); refit meta per split
    for i,j in skf:
        m=LogisticRegression(C=C,max_iter=8000).fit(X[i],y[i]); o[j]=norm(W*m.predict_proba(X[j])+(1-W)*xg_oof[j])
    vals.append(log_loss(y,o))
print('meta-seed vals',vals,'mean',np.mean(vals),flush=True)
mlr=LogisticRegression(C=C,max_iter=8000).fit(X,y); lr_t=norm(mlr.predict_proba(XT))
g=XGBClassifier(n_estimators=200,learning_rate=0.03,max_depth=3,subsample=0.8,colsample_bytree=0.6,reg_lambda=1.0,objective='multi:softprob',tree_method='hist',n_jobs=2,random_state=42,verbosity=0).fit(X,y); xg_t=norm(g.predict_proba(XT))
test=norm(W*lr_t+(1-W)*xg_t); test=np.clip(test,1e-9,1); test=test/test.sum(1,keepdims=True)
sub=pd.DataFrame({'id':te['id'],'EAP':test[:,0],'HPL':test[:,1],'MWS':test[:,2]})
os.makedirs(FS,exist_ok=True); sub.to_csv(f'{FS}/submission.csv',index=False)
np.save(f'{OUT}/final_oof.npy',oof); np.save(f'{OUT}/final_test.npy',test)
pd.DataFrame({'id':te['id'],'EAP':norm(lr_t)[:,0],'HPL':norm(lr_t)[:,1],'MWS':norm(lr_t)[:,2]}).to_csv(f'{OUT}/submission_lr_fallback.csv',index=False)
json.dump({'meta':'LR(C=0.1) + XGB blend on [log p, p], w=0.8/0.2','w':W,'C':C,'cv':cv,'cv_lr':cv_lr,'cv_xgb':cv_xg,
           'meta_seed_mean':float(np.mean(vals)),'meta_seed_vals':vals,'n':len(names),'models':names,
           'note':'base-only: excludes partial/stack/blend/candidate/nested/meta/final/opt; maxll 0.75; dedup corr 0.998'},
          open(f'{OUT}/final_recipe.json','w'),indent=2)
print('wrote submission',sub.shape,flush=True)
