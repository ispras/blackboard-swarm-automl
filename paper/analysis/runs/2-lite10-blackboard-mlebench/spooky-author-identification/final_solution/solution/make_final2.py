import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os, json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from stack_all import collect, folds, y, te
EPS=1e-15
ROOT='/download_job_code/payload/sandbox'
OUT=f'{ROOT}/workspaces/canopus_kappa'; FS=f'{ROOT}/final_solution'
DERIVED=('partial','stack','blend','candidate','nested','meta_')
MAXLL=0.60; THR=0.998

def normalize(p):
    p=np.clip(p,EPS,1); return p/p.sum(1,keepdims=True)

def feats(P, kind):
    L=np.log(P)
    if kind=='log': return L.transpose(1,0,2).reshape(len(P[0]),-1)
    return np.concatenate([L,P],axis=2).transpose(1,0,2).reshape(len(P[0]),-1)

def stack_cv(O, kind, C):
    X=feats(O,kind); oof=np.zeros((len(y),3))
    for i,j in folds:
        m=LogisticRegression(C=C,max_iter=5000); m.fit(X[i],y[i]); oof[j]=m.predict_proba(X[j])
    return log_loss(y,oof),oof

def fit_predict(O,T,kind,C):
    X=feats(O,kind); XT=feats(T,kind)
    m=LogisticRegression(C=C,max_iter=5000); m.fit(X,y)
    return normalize(m.predict_proba(XT))

def base_pool():
    names,O,T=collect(); keep=[]
    for k,n in enumerate(names):
        t=n.split('/')[-1].lower()
        if any(x in t for x in DERIVED): continue
        if not np.isfinite(O[k]).all() or not np.isfinite(T[k]).all(): continue
        if log_loss(y,normalize(O[k]))>MAXLL: continue
        keep.append(k)
    ind={k:log_loss(y,normalize(O[k])) for k in keep}; order=sorted(keep,key=lambda k:ind[k]); sel=[]
    for k in order:
        if any(np.corrcoef(O[k].ravel(),O[j].ravel())[0,1]>THR for j in sel): continue
        sel.append(k)
    sel=sorted(sel)
    return [names[k] for k in sel], np.stack([normalize(O[k]) for k in sel]), np.stack([normalize(T[k]) for k in sel])

if __name__=='__main__':
    names,O,T=base_pool()
    print('n base',len(names),flush=True)
    results={}
    for kind in ['log','both']:
        best=(9,None,None)
        for C in [0.03,0.05,0.1,0.2,0.3]:
            s,oof=stack_cv(O,kind,C); print(f'{kind} C={C}: {s:.5f}',flush=True)
            if s<best[0]: best=(s,C,oof)
        results[kind]=best
    kind=min(results,key=lambda k:results[k][0]); s,C,oof=results[kind]
    print('CHOSEN',kind,'C',C,'CV',s,flush=True)
    # repeated CV with different meta seeds for robustness
    from sklearn.model_selection import StratifiedKFold
    for seed in [0,1,7]:
        skf=StratifiedKFold(5,shuffle=True,random_state=seed); X=feats(O,kind); o=np.zeros((len(y),3))
        for i,j in skf.split(X,y):
            m=LogisticRegression(C=C,max_iter=5000); m.fit(X[i],y[i]); o[j]=m.predict_proba(X[j])
        print('meta-seed',seed,round(log_loss(y,o),5),flush=True)
    test=fit_predict(O,T,kind,C)
    test=np.clip(test,1e-9,1); test=test/test.sum(1,keepdims=True)
    sub=pd.DataFrame({'id':te['id'],'EAP':test[:,0],'HPL':test[:,1],'MWS':test[:,2]})
    os.makedirs(FS,exist_ok=True); sub.to_csv(f'{FS}/submission.csv',index=False)
    np.save(f'{OUT}/final_oof.npy',oof); np.save(f'{OUT}/final_test.npy',test)
    json.dump({'kind':kind,'C':C,'cv':s,'n':len(names),'models':names,
               'all_variants':{k:{'cv':results[k][0],'C':results[k][1]} for k in results}},
              open(f'{OUT}/final_recipe.json','w'),indent=2)
    print('wrote',f'{FS}/submission.csv',sub.shape,flush=True)
