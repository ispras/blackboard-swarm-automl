import os, sys, json, math, datetime, time
import numpy as np
from functools import partial
print=partial(__import__('builtins').print, flush=True)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA='/download_job_code/payload/sandbox/data'
OUT='/download_job_code/payload/sandbox/workspaces/altair_theta'
RIGEL='/download_job_code/payload/sandbox/workspaces/rigel_omega'
t0=time.time()
labels={}
with open(f'{DATA}/essential_data/rec_labels_test_hidden.txt') as f:
    next(f)
    for line in f:
        p=line.strip().split(','); rid=int(p[0]); rest=[x for x in p[1:] if x!='']
        labels[rid]=('?' if (rest and rest[0]=='?') else set(int(x) for x in rest))
fold={}
with open(f'{DATA}/essential_data/CVfolds_2.txt') as f:
    next(f)
    for line in f:
        r,fo=line.strip().split(','); fold[int(r)]=int(fo)
fn={}
with open(f'{DATA}/essential_data/rec_id2filename.txt') as f:
    next(f)
    for line in f:
        r,f2=line.strip().split(','); fn[int(r)]=f2
train=sorted(r for r in fold if fold[r]==0); test=sorted(r for r in fold if fold[r]==1)
Y=np.zeros((len(train),19))
for i,r in enumerate(train):
    if isinstance(labels.get(r),set):
        for c in labels[r]: Y[i,c]=1
stations=sorted(set(fn[r].split('_')[0] for r in fn)); si={s:i for i,s in enumerate(stations)}
dates=sorted(set(fn[r].split('_')[1] for r in fn)); di={s:i for i,s in enumerate(dates)}
def meta(r):
    p=fn[r].split('_'); st=p[0]; date=p[1]; t=p[2]
    y=int(date[:4]); mo=int(date[4:6]); d=int(date[6:8])
    mins=int(t[:2])*60+int(t[2:4])+int(t[4:6])/60
    v=[1.0 if i==si[st] else 0.0 for i in range(len(stations))]
    v+=[1.0 if i==di[date] else 0.0 for i in range(len(dates))]
    v+=[1.0 if y==2010 else 0.0]
    v+=[math.sin(2*math.pi*mins/1440),math.cos(2*math.pi*mins/1440),mins/1440.0]
    return np.array(v,dtype=np.float32)
z=np.load(f'{OUT}/ast_tokens.npz',allow_pickle=True); toks=z['tokens']; recs=list(z['recs']); idx={r:i for i,r in enumerate(recs)}
az=np.load(f'{RIGEL}/ast_feats.npz',allow_pickle=True); arecs=list(az['recs']); aidx={r:i for i,r in enumerate(arecs)}; aemb=az['emb']
bz=np.load(f'{RIGEL}/birdnet_emb.npz',allow_pickle=True); brecs=list(bz['recs']); bidx={r:i for i,r in enumerate(brecs)}; bmean=bz['mean']; bmax=bz['max']
def feats(rids):
    N=len(rids); pmax=np.zeros((N,768),np.float32); astr=np.zeros((N,768),np.float32)
    bm=np.zeros((N,1024),np.float32); bx=np.zeros((N,1024),np.float32)
    for j,r in enumerate(rids):
        pmax[j]=toks[idx[r]].astype(np.float32).max(0)
        astr[j]=aemb[aidx[r]]; bm[j]=bmean[bidx[r]]; bx[j]=bmax[bidx[r]]
    return np.hstack([astr,pmax,bm,bx,np.vstack([meta(r) for r in rids])])
Xtr=feats(train); Xte=feats(test); print("dims",Xtr.shape,flush=True)
def mg(P):
    au=[roc_auc_score(Y[:,c],P[:,c]) for c in range(19) if 0<Y[:,c].sum()<len(Y)]
    return float(np.mean(au)), float(roc_auc_score(Y.ravel(),P.ravel()))
def clf(): return LogisticRegression(C=0.1,max_iter=1000,class_weight='balanced')
oof=np.zeros((len(train),19))
for seed in (0,1,2):
    for tr,va in KFold(5,shuffle=True,random_state=seed).split(train):
        for c in range(19):
            y=Y[tr,c]
            if y.sum()==0: continue
            if y.sum()==len(y): oof[va,c]+=1; continue
            sc=StandardScaler().fit(Xtr[tr])
            m=clf().fit(sc.transform(Xtr[tr]),y)
            oof[va,c]+=m.predict_proba(sc.transform(Xtr[va]))[:,1]
oof/=3.0; m,g=mg(oof)
P=np.zeros((len(test),19))
for c in range(19):
    y=Y[:,c]
    if y.sum()==0: continue
    sc=StandardScaler().fit(Xtr); mm=clf().fit(sc.transform(Xtr),y)
    P[:,c]=mm.predict_proba(sc.transform(Xte))[:,1]
np.savez(f'{OUT}/oof_ast_pmax_bnemb_meta.npz',rec_ids=np.array(train),probs=oof,test_rec_ids=np.array(test),test_probs=P)
json.dump({'name':'ast_pmax_bnemb_meta','macro':m,'global':g,'C':0.1},open(f'{OUT}/oof_ast_pmax_bnemb_meta.json','w'),indent=2)
print(f'ast_pmax_bnemb_meta macro={m:.4f} global={g:.4f} total={time.time()-t0:.1f}s',flush=True)
