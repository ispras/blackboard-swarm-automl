import os, json, math
import numpy as np
from functools import partial
print=partial(__import__('builtins').print, flush=True)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
DATA='/download_job_code/payload/sandbox/data'; OUT='/download_job_code/payload/sandbox/workspaces/altair_theta'
labels={}
for line in open(f'{DATA}/essential_data/rec_labels_test_hidden.txt').read().splitlines()[1:]:
    p=line.split(','); rid=int(p[0]); rest=[x for x in p[1:] if x!='']
    labels[rid]=('?' if (rest and rest[0]=='?') else set(int(x) for x in rest))
fold={}
for line in open(f'{DATA}/essential_data/CVfolds_2.txt').read().splitlines()[1:]:
    r,fo=line.split(','); fold[int(r)]=int(fo)
fn={}
for line in open(f'{DATA}/essential_data/rec_id2filename.txt').read().splitlines()[1:]:
    r,f2=line.split(','); fn[int(r)]=f2
train=sorted(r for r in fold if fold[r]==0); test=sorted(r for r in fold if fold[r]==1)
Y=np.zeros((len(train),19))
for i,r in enumerate(train):
    if isinstance(labels.get(r),set):
        for c in labels[r]: Y[i,c]=1
stations=sorted(set(fn[r].split('_')[0] for r in fn)); si={s:i for i,s in enumerate(stations)}
dates=sorted(set(fn[r].split('_')[1] for r in fn)); di={s:i for i,s in enumerate(dates)}
def meta(r):
    p=fn[r].split('_'); st=p[0]; date=p[1]; t=p[2]
    y=int(date[:4]); mo=int(date[4:6]); d=int(date[6:8]); mins=int(t[:2])*60+int(t[2:4])+int(t[4:6])/60
    v=[1.0 if i==si[st] else 0.0 for i in range(len(stations))]+[1.0 if i==di[date] else 0.0 for i in range(len(dates))]
    v+=[1.0 if y==2010 else 0.0]+[math.sin(2*math.pi*mins/1440),math.cos(2*math.pi*mins/1440),mins/1440.0]
    return np.array(v,dtype=np.float32)
d=np.load(f'{OUT}/emb_convnext_birdset.npz',allow_pickle=True); rc=list(d['recs']); ridx={r:i for i,r in enumerate(rc)}
W=d['windows']; own=d['owner']; wins={i:np.where(own==i)[0] for i in range(len(rc))}
def feats_windows(rec_idxs):
    X=[]; bag=[]
    for b,i in enumerate(rec_idxs):
        m=meta(rc[i])
        for w in wins[i]: X.append(np.concatenate([W[w].astype(np.float32),m])); bag.append(b)
    return np.array(X),np.array(bag)
Xtr,bagtr=feats_windows([ridx[r] for r in train]); Xte,bagte=feats_windows([ridx[r] for r in test])
sc=StandardScaler().fit(Xtr); Ztr=sc.transform(Xtr); Zte=sc.transform(Xte)
Ptr=np.zeros((Xtr.shape[0],19)); Pte=np.zeros((Xte.shape[0],19))
for c in range(19):
    yw=Y[bagtr,c]
    if yw.sum()==0: continue
    m=LogisticRegression(C=0.1,max_iter=1000,class_weight='balanced').fit(Ztr,yw)
    Ptr[:,c]=m.predict_proba(Ztr)[:,1]; Pte[:,c]=m.predict_proba(Zte)[:,1]
def agg(P,bag,n):
    out=np.full((n,19),np.nan)
    for b in np.unique(bag): out[b]=P[bag==b].max(0)
    return out
# OOF from saved
oof=np.load(f'{OUT}/oof_miml_cn_max.npz')['probs']
T=agg(Pte,bagte,len(test))
np.savez(f'{OUT}/oof_miml_cn_max.npz',rec_ids=np.array(train),probs=oof,test_rec_ids=np.array(test),test_probs=T)
from sklearn.metrics import roc_auc_score
au=[roc_auc_score(Y[:,c],oof[:,c]) for c in range(19) if 0<Y[:,c].sum()<len(Y)]
print("miml_cn_max macro",round(np.mean(au),4),"global",round(roc_auc_score(Y.ravel(),oof.ravel()),4),flush=True)
json.dump({'name':'miml_cn_max'},open(f'{OUT}/oof_miml_cn_max.json','w'))
