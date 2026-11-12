import sys, json, csv, pickle
sys.path.insert(0,'../mirfak_chi')
import numpy as np, lightgbm as lgb
from v7_pipeline import build_features, load_extra, load_mass, norm, PUNCT
BASE='.'; FOM='../fomalhaut_zeta'; VER='../mirfak_chi'
truth=[json.loads(l) for l in open(f'{BASE}/val_truth.jsonl')]
vt={int(r['id']):r['sentence'] for r in csv.DictReader(open(f'{BASE}/val_test.csv'))}
tg={d['id']:d['gap'] for d in truth}
fom=load_mass(f'{FOM}/fomalhaut_val_gap_probs20m.jsonl')
mlm={}
for l in open(f'{BASE}/mlm_val20k.jsonl'):
    d=json.loads(l); mlm[d['id']]=d['pred']
causal={}
for l in open(f'{FOM}/causal_surp_val.jsonl'):
    d=json.loads(l); causal[d['id']]=d['surp']
v1=load_extra(f'{VER}/verif_gaps_val.jsonl'); v2=load_extra(f'{VER}/verif_ft_gaps_val.jsonl')
v3=load_extra(f'{BASE}/val_verif_large.jsonl')
X=[];Y=[];rid=[];meta=[]
for d in truth:
    i=d['id'];T=vt[i].split(' ')
    rows,gaps=build_features(T,mlm[i],fom[i],causal.get(i,[]),v1.get(i,{}),v2.get(i,{}),v3.get(i,{}))
    for r,g in zip(rows,gaps): X.append(r);Y.append(1 if g==tg[i] else 0);rid.append(i)
    meta.append((i,gaps))
X=np.array(X,dtype=np.float32);Y=np.array(Y,dtype=np.int8);rid=np.array(rid)
tr=rid<5000
clf=lgb.LGBMClassifier(objective='binary',num_leaves=31,learning_rate=0.05,n_estimators=400,min_child_samples=50,n_jobs=4,verbose=-1)
clf.fit(X[tr],Y[tr]); s=clf.predict_proba(X)[:,1]
k=0;t1=t3=0;n=0
with open('val_gaps_v7.jsonl','w') as fo:
    for i,gaps in meta:
        sc=s[k:k+len(gaps)];k+=len(gaps)
        order=np.argsort(-sc);pg=int(np.array(gaps)[order[0]])
        mg=float(sc[order[0]]-sc[order[1]]) if len(order)>1 else 99.0
        lg=np.log(np.clip(sc,1e-9,1-1e-9)/np.clip(1-sc,1e-9,1-1e-9));e=np.exp(lg-lg.max());post=e/e.sum()
        fo.write(json.dumps({'id':i,'gap':pg,'margin':mg,'p1':float(post[order[0]]),'gap_probs':[[int(gaps[j]),float(post[j])] for j in order[:10]]})+'\n')
        if i>=5000:
            r=list(np.array(gaps)[order]).index(tg[i]); t1+=r==0;t3+=r<3;n+=1
print('VAL v7 heldout top1',round(t1/n,4),'top3',round(t3/n,4),'n',n)
