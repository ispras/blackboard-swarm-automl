import os, json, math, time
import numpy as np
from functools import partial
print=partial(__import__('builtins').print, flush=True)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
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
W=d['windows']; own=d['owner']
rec_train=[ridx[r] for r in train]; rec_test=[ridx[r] for r in test]
# per-rec window indices
wins={i:np.where(own==i)[0] for i in range(len(rc))}
def feats_windows(rec_idxs):
    X=[]; bag=[]
    for b,i in enumerate(rec_idxs):
        ids=wins[i]
        m=meta(rc[i])
        for w in ids:
            X.append(np.concatenate([W[w].astype(np.float32), m])); bag.append(b)
    return np.array(X), np.array(bag)
def agg(p, bag, n, how='max'):
    out=np.full((n,19), np.nan)
    for b in np.unique(bag):
        rows=p[bag==b]
        if how=='max': out[b]=rows.max(0)
        elif how=='mean': out[b]=rows.mean(0)
        else: out[b]=np.log(np.exp(rows).mean(0))
    return out
def mg(P):
    au=[roc_auc_score(Y[:,c],P[:,c]) for c in range(19) if 0<Y[:,c].sum()<len(Y)]
    return float(np.mean(au)), float(roc_auc_score(Y.ravel(),P.ravel()))
Xtr,bagtr=feats_windows(rec_train); Xte,bagte=feats_windows(rec_test)
print("windows",Xtr.shape,Xte.shape,flush=True)
def run(how,C=0.1):
    oof=np.zeros((len(train),19))
    for seed in (0,1,2):
        for tr,va in KFold(5,shuffle=True,random_state=seed).split(train):
            # window mask for train recs
            mtr=np.isin(bagtr,tr); mva=np.isin(bagtr,va)
            sc=StandardScaler().fit(Xtr[mtr])
            Ztr=sc.transform(Xtr[mtr]); Zva=sc.transform(Xtr[mva])
            btr=bagtr[mtr]; bva=bagtr[mva]
            Pva=np.zeros((len(Xtr[mva]),19))
            for c in range(19):
                yc=Y[tr,c]
                if yc.sum()==0: continue
                if yc.sum()==len(yc): Pva[:,c]=1; continue
                yw=Y[btr,c]  # window labels (global indices)
                if yw.sum()==0: continue
                m=LogisticRegression(C=C,max_iter=1000,class_weight='balanced').fit(Ztr,yw)
                Pva[:,c]=m.predict_proba(Zva)[:,1]
            A=agg(Pva,bva,len(train),how)
            oof[va]+=A[va]/3.0
    return oof
for how in ('max','mean'):
    for C in (0.1,):
        oof=run(how,C); m,g=mg(oof); print(f"MIML {how} C={C} macro={m:.4f} global={g:.4f}",flush=True)
        np.savez(f'{OUT}/oof_miml_cn_{how}.npz',rec_ids=np.array(train),probs=oof,Y=Y)
