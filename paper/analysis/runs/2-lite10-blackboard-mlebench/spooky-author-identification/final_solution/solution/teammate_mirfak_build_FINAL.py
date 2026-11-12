import numpy as np, pandas as pd, os, glob, warnings, json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')
ROOT='/download_job_code/payload/sandbox'; OUT=f'{ROOT}/workspaces/mirfak_pi'
tr=pd.read_csv(f'{ROOT}/data/train.csv'); te=pd.read_csv(f'{ROOT}/data/test.csv')
y=tr['author'].map({'EAP':0,'HPL':1,'MWS':2}).values
fold_id=np.load(f'{ROOT}/workspaces/bellatrix_tau/fold_id.npy'); N=len(y); NT=len(te)
EXCL=('partial','stack','blend','candidate','nested','meta','opt','final')
models={}
for ws in ['bellatrix_tau','canopus_kappa','mirfak_pi']:
    for f in sorted(glob.glob(f'{ROOT}/workspaces/{ws}/oof_*.npy')):
        name=os.path.basename(f)[4:-4]
        if any(e in name for e in EXCL): continue
        p=np.load(f); tp=f'{ROOT}/workspaces/{ws}/test_{name}.npy'
        if p.shape!=(N,3) or not os.path.exists(tp): continue
        t=np.load(tp)
        if t.shape!=(NT,3): continue
        p=np.clip(p,1e-9,1); p/=p.sum(1,keepdims=True); t=np.clip(t,1e-9,1); t/=t.sum(1,keepdims=True)
        k=f'{ws[:2]}:{name}'
        if k not in models: models[k]=(p,t)
sc={k:log_loss(y,v[0],labels=[0,1,2]) for k,v in models.items()}
models={k:v for k,v in models.items() if sc[k]<=0.75}
names=sorted(models,key=lambda k:sc[k]); keep=[]
for k in names:
    lk=np.log(models[k][0]).ravel(); dup=False
    for j in keep:
        if np.corrcoef(lk,np.log(models[j][0]).ravel())[0,1]>0.998: dup=True;break
    if not dup: keep.append(k)
names=keep
assert not any(any(e in n for e in ('meta','final','stack','blend','candidate','nested','partial','opt')) for n in names)
P=np.concatenate([models[n][0] for n in names],1); PT=np.concatenate([models[n][1] for n in names],1)
X=np.hstack([np.log(P),P]); XT=np.hstack([np.log(PT),PT])
def mk(k):
    if k=='lr': return LogisticRegression(C=0.05,max_iter=4000)
    if k=='hgb': return HistGradientBoostingClassifier(max_iter=200,learning_rate=0.03,max_leaf_nodes=15,l2_regularization=1.0,random_state=0)
    return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.5,reg_lambda=1.0,n_jobs=2,eval_metric='mlogloss',verbosity=0,random_state=0)
W={'lr':0.60,'hgb':0.20,'xgb':0.20}
oofs={}; tests={}
for k in W:
    o=np.zeros((N,3))
    for f in range(5):
        itr=np.where(fold_id!=f)[0]; iva=np.where(fold_id==f)[0]
        o[iva]=mk(k).fit(X[itr],y[itr]).predict_proba(X[iva])
    oofs[k]=o; tests[k]=mk(k).fit(X,y).predict_proba(XT)
    print(k,'cv',round(log_loss(y,o,labels=[0,1,2]),5),flush=True)
oof=sum(W[k]*oofs[k] for k in W); pt=sum(W[k]*tests[k] for k in W)
cv=log_loss(y,oof,labels=[0,1,2]); print('FINAL 3-way CV',cv,flush=True)
np.save(f'{OUT}/oof_FINAL_honest.npy',oof); np.save(f'{OUT}/test_FINAL_honest.npy',pt)
sub=pd.DataFrame({'id':te['id'],'EAP':pt[:,0],'HPL':pt[:,1],'MWS':pt[:,2]})
sub.to_csv(f'{OUT}/sub_FINAL_honest.csv',index=False)
json.dump({'cv':float(cv),'weights':W,'n':len(names),'models':names,'oof_ll':{k:float(log_loss(y,oofs[k],labels=[0,1,2])) for k in W}},open(f'{OUT}/FINAL_results.json','w'),indent=2)
print('saved rows',len(sub),'sum1 ok',np.allclose(sub[['EAP','HPL','MWS']].sum(1),1))
