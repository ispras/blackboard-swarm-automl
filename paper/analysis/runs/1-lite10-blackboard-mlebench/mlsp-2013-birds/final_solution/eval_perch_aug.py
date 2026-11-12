import os, json, math
import numpy as np
from functools import partial
print=partial(__import__('builtins').print, flush=True)
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
DATA='/download_job_code/payload/sandbox/data'; OUT='/download_job_code/payload/sandbox/workspaces/altair_theta'; RIGEL='/download_job_code/payload/sandbox/workspaces/rigel_omega'
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
def load_npz(path,key):
    d=np.load(path,allow_pickle=True); rc=list(d['recs']); return {r:d[key][i] for i,r in enumerate(rc)}
pm=load_npz(f'{RIGEL}/emb_perch.npz','X'); px=load_npz(f'{RIGEL}/emb_perchmax.npz','X')
aug=np.load(f'{OUT}/emb_perch_aug.npz',allow_pickle=True); arc=list(aug['recs']); aidx={r:i for i,r in enumerate(arc)}; A=aug['aug']
def orig(r): return np.concatenate([pm[r],px[r],meta(r)]).astype(np.float32)
def augv(r,k): return np.concatenate([A[aidx[r],k],meta(r)]).astype(np.float32)
Otr=np.vstack([orig(r) for r in train]); Ote=np.vstack([orig(r) for r in test])
def mg(P):
    au=[roc_auc_score(Y[:,c],P[:,c]) for c in range(19) if 0<Y[:,c].sum()<len(Y)]
    return float(np.mean(au)), float(roc_auc_score(Y.ravel(),P.ravel()))
def run(C, tta=False):
    oof=np.zeros((len(train),19))
    for seed in (0,1,2):
        for tr,va in KFold(5,shuffle=True,random_state=seed).split(train):
            Ztr=[Otr[tr]]+[np.vstack([augv(train[i],k) for i in tr]) for k in range(A.shape[1])]
            Ztr=np.vstack(Ztr); ytr=np.tile(Y[tr],(1+A.shape[1],1))
            sc=StandardScaler().fit(Ztr); Ztr=sc.transform(Ztr)
            Pva=np.zeros((len(va),19))
            for c in range(19):
                yc=ytr[:,c]
                if Y[tr,c].sum()==0: continue
                if Y[tr,c].sum()==len(tr): Pva[:,c]=1; continue
                m=LogisticRegression(C=C,max_iter=1000,class_weight='balanced').fit(Ztr,yc)
                pv=m.predict_proba(sc.transform(Otr[va]))[:,1]
                if tta:
                    for k in range(A.shape[1]):
                        pv+=m.predict_proba(sc.transform(np.vstack([augv(train[i],k) for i in va])))[:,1]
                    pv/= (1+A.shape[1])
                Pva[:,c]=pv
            oof[va]+=Pva/3.0
    return oof
for C in (0.1,0.3):
    for tta in (False,True):
        oof=run(C,tta); m,g=mg(oof); print(f"perch-aug C={C} tta={tta} macro={m:.4f} global={g:.4f}",flush=True)
# export best (C=0.3 tta) with test preds
def full_test(C,tta):
    Ztr=[Otr]+[np.vstack([augv(r,k) for r in train]) for k in range(A.shape[1])]
    Ztr=np.vstack(Ztr); ytr=np.tile(Y,(1+A.shape[1],1)); sc=StandardScaler().fit(Ztr); Ztr=sc.transform(Ztr)
    Pte=np.zeros((len(test),19))
    for c in range(19):
        yc=ytr[:,c]
        if yc.sum()==0: continue
        m=LogisticRegression(C=C,max_iter=1000,class_weight='balanced').fit(Ztr,yc)
        pv=m.predict_proba(sc.transform(Ote))[:,1]
        if tta:
            for k in range(A.shape[1]):
                pv+=m.predict_proba(sc.transform(np.vstack([augv(r,k) for r in test])))[:,1]
            pv/=(1+A.shape[1])
        Pte[:,c]=pv
    return Pte
oof=run(0.3,True); Pte=full_test(0.3,True)
np.savez(f'{OUT}/oof_perch_aug.npz',rec_ids=np.array(train),probs=oof,test_rec_ids=np.array(test),test_probs=Pte)
json.dump({'name':'perch_aug_tta'},open(f'{OUT}/oof_perch_aug.json','w'))
print("exported perch_aug",flush=True)
