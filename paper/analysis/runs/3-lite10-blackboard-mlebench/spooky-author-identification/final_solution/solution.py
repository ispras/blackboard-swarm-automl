import numpy as np, pandas as pd, glob, os, json, warnings, sys, time
warnings.filterwarnings('ignore')
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

ROOT='/download_job_code/payload/sandbox'
DATA=f'{ROOT}/data'; OUT=f'{ROOT}/final_solution'
TR=pd.read_csv(f'{DATA}/train.csv'); TE=pd.read_csv(f'{DATA}/test.csv')
CLASSES=['EAP','HPL','MWS']; y=TR['author'].map({c:i for i,c in enumerate(CLASSES)}).values
os.makedirs(OUT,exist_ok=True)

# ---- discover base predictions (any agent workspace) ----
pairs={}
for f in glob.glob(f'{ROOT}/workspaces/*/preds/oof_*.npy')+glob.glob(f'{ROOT}/workspaces/*/oof_*.npy'):
    name=os.path.basename(f)[4:-4]; d=os.path.dirname(f); tf=os.path.join(d,'test_'+name+'.npy')
    if not os.path.exists(tf): continue
    if any(s in name.lower() for s in ['stack','cand','blend','submission','meta']): continue
    agent=d.split('/workspaces/')[1].split('/')[0]; key=f'{agent}:{name}'
    o=np.load(f); t=np.load(tf)
    if o.shape!=(len(TR),3) or t.shape!=(len(TE),3): continue
    if not np.isfinite(o).all() or not np.isfinite(t).all(): continue
    # reject incomplete/partial prediction files (unfilled folds are all-zero rows)
    if (o.sum(1)<0.5).any() or (t.sum(1)<0.5).any(): continue
    if ((o.max(1)-o.min(1))<1e-9).mean()>0.01: continue
    o=np.clip(o,1e-12,1); o=o/o.sum(1,keepdims=True)
    t=np.clip(t,1e-12,1); t=t/t.sum(1,keepdims=True)
    pairs[key]=(o,t)
scores={k:log_loss(y,o) for k,(o,t) in pairs.items()}
# keep models below cutoff (weak models add little / hurt); tunable
CUT=float(os.environ.get('CUT','0.65'))
# models shown neutral/harmful by multi-seed nested CV (agrees with independent achernar audit)
EXCLUDE=[x for x in os.environ.get('EXCLUDE','dv3l_s555,bertlargec_full').split(',') if x]
keys=[k for k in sorted(scores,key=scores.get) if scores[k]<CUT and not any(x in k for x in EXCLUDE)]
print(f'pool={len(pairs)} kept={len(keys)} cutoff={CUT}',flush=True)
for k in keys: print(f'  {scores[k]:.4f} {k}')

O=np.stack([pairs[k][0] for k in keys]); T=np.stack([pairs[k][1] for k in keys])
# stacker features: log(p) AND raw p (better than log-only on nested CV)
Z=np.concatenate([np.log(O),O],axis=2).transpose(1,0,2).reshape(len(TR),-1)
Zt=np.concatenate([np.log(T),T],axis=2).transpose(1,0,2).reshape(len(TE),-1)
C=float(os.environ.get('STACK_C','0.01'))

# ---- honest nested-CV estimate of stack ----
lls=[]
for r in range(3):
    skf=StratifiedKFold(5,shuffle=True,random_state=100+r); oof=np.zeros((len(TR),3))
    for ti,vi in skf.split(Z,y):
        m=LogisticRegression(C=C,max_iter=1000).fit(Z[ti],y[ti]); oof[vi]=m.predict_proba(Z[vi])
    lls.append(log_loss(y,oof))
nested=float(np.mean(lls)); nested_sd=float(np.std(lls))
print(f'STACK nested OOF logloss = {nested:.4f} +- {nested_sd:.4f}',flush=True)

# ---- final: bagged stackers fit on random 80% subsets, average test preds ----
rng=np.random.RandomState(7); test_pred=np.zeros((len(TE),3)); nbag=20
for b in range(nbag):
    idx=rng.permutation(len(TR))[:int(0.8*len(TR))]
    m=LogisticRegression(C=C,max_iter=1000).fit(Z[idx],y[idx])
    test_pred+=m.predict_proba(Zt)
test_pred/=nbag
test_pred=np.clip(test_pred,1e-15,1)
test_pred=test_pred/test_pred.sum(1,keepdims=True)

sub=pd.DataFrame({'id':TE['id'].values,'EAP':test_pred[:,0],'HPL':test_pred[:,1],'MWS':test_pred[:,2]})
sub.to_csv(f'{OUT}/submission.csv',index=False)
np.save(f'{OUT}/test_probs.npy',test_pred)
json.dump({'nested_oof_logloss':nested,'nested_sd':nested_sd,'C':C,'cutoff':CUT,
           'n_models':len(keys),'models':keys,
           'single_model_oof':{k:scores[k] for k in keys}},
          open(f'{OUT}/stack_meta.json','w'),indent=2)
print('wrote',f'{OUT}/submission.csv',sub.shape,flush=True)
print(sub.head())
