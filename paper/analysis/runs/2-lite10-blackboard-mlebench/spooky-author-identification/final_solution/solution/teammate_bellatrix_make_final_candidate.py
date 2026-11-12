import numpy as np, pandas as pd, os, glob, json, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')
ROOT='/download_job_code/payload/sandbox'; WORK=f'{ROOT}/workspaces'; OUT=f'{WORK}/bellatrix_tau'
tr=pd.read_csv(f'{ROOT}/data/train.csv'); te=pd.read_csv(f'{ROOT}/data/test.csv')
y=tr['author'].map({'EAP':0,'HPL':1,'MWS':2}).values
fold_id=np.load(f'{OUT}/fold_id.npy'); N,M=len(y),len(te)
EXCL=('stack','blend','optblend','mystack','crossstack','partial','chosen','sub','candidate','nested','meta','final','lr3mix','lrhgb','3way')
TFK=('dbl','deberta','roberta','electra','bert')

models={}
for ws in sorted(os.listdir(WORK)):
    d=f'{WORK}/{ws}'
    if not os.path.isdir(d): continue
    for f in sorted(glob.glob(f'{d}/oof_*.npy')):
        name=os.path.basename(f)[4:-4]
        if any(e in name for e in EXCL): continue
        tp=f'{d}/test_{name}.npy'
        if not os.path.exists(tp): continue
        p=np.load(f); t=np.load(tp)
        if p.shape!=(N,3) or t.shape!=(M,3): continue
        if not np.isfinite(p).all() or not np.isfinite(t).all(): continue
        p=np.clip(p,1e-9,1); p/=p.sum(1,keepdims=True); t=np.clip(t,1e-9,1); t/=t.sum(1,keepdims=True)
        ll=log_loss(y,p,labels=[0,1,2])
        if ll>0.75: continue
        models[f'{ws}:{name}']=(p,t,ll)
keys=sorted(models,key=lambda k:models[k][2]); kept=[]
for k in keys:
    if any(np.corrcoef(models[k][0].ravel(),models[kk][0].ravel())[0,1]>0.998 for kk in kept): continue
    kept.append(k)
tf=[k for k in kept if any(s in k.split(':')[1] for s in TFK)]
print(f'base models loaded {len(models)}, kept {len(kept)}, transformers {tf}',flush=True)
LP=lambda P: np.log(np.clip(P,1e-6,1))
XB=np.concatenate([LP(models[k][0]) for k in kept],axis=1)
XBT=np.concatenate([LP(models[k][1]) for k in kept],axis=1)
P=np.concatenate([models[k][0] for k in kept],axis=1); PT=np.concatenate([models[k][1] for k in kept],axis=1)

def cv_variant(X,XT,C):
    oof=np.zeros((N,3))
    for f in range(5):
        itr=np.where(fold_id!=f)[0];iva=np.where(fold_id==f)[0]
        m=LogisticRegression(C=C,max_iter=4000,multi_class='multinomial');m.fit(X[itr],y[itr])
        oof[iva]=m.predict_proba(X[iva])
    m=LogisticRegression(C=C,max_iter=4000,multi_class='multinomial');m.fit(X,y)
    return oof,m.predict_proba(XT)

variants=[
    ('S1a_logprob',XB,XBT,0.1),
    ('S1b_logprob+prob',np.hstack([XB,P]),np.hstack([XBT,PT]),0.1),
]
results=[]
for name,X,XT,C in variants:
    o,t=cv_variant(X,XT,C);sc=log_loss(y,o,labels=[0,1,2])
    results.append((sc,name,o,t));print(f'{name}: {sc:.5f}',flush=True)
results.sort()
sc,name,O,T=results[0]
print('CHOSEN',name,sc,flush=True)
sub=pd.DataFrame({'id':te['id'],'EAP':T[:,0],'HPL':T[:,1],'MWS':T[:,2]})
sub.to_csv(f'{OUT}/meta_candidate_submission.csv',index=False)
np.save(f'{OUT}/meta_candidate_oof.npy',O);np.save(f'{OUT}/meta_candidate_test.npy',T)
json.dump({'chosen':name,'cv':float(sc),'n_base':len(kept),'n_tf':len(tf),'tf':tf,
           'variants':{n:float(s) for s,n,_,_ in results},'models':kept},
          open(f'{OUT}/candidate_results.json','w'),indent=2)
print('WROTE',f'{OUT}/meta_candidate_submission.csv',sub.shape)
