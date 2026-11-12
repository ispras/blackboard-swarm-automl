import os, json, math, time
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
z=np.load(f'{OUT}/ast_tokens.npz',allow_pickle=True); toks=z['tokens']; recs=list(z['recs']); idx={r:i for i,r in enumerate(recs)}
def load_npz(path,key):
    d=np.load(path,allow_pickle=True); rc=list(d['recs']); return {r:d[key][i] for i,r in enumerate(rc)}
clap=load_npz(f'{OUT}/emb_clap.npz','X')
perch=load_npz(f'{RIGEL}/emb_perch.npz','X'); perchmax=load_npz(f'{RIGEL}/emb_perchmax.npz','X')
def feats(rids):
    N=len(rids); cl=np.zeros((N,512),np.float32); pm=np.zeros((N,768),np.float32); pe=np.zeros((N,1536),np.float32); px=np.zeros((N,1536),np.float32)
    for j,r in enumerate(rids):
        cl[j]=clap[r]; pm[j]=toks[idx[r]].astype(np.float32).max(0); pe[j]=perch[r]; px[j]=perchmax[r]
    return dict(clap=cl,pmax=pm,perch=pe,perchmax=px,meta=np.vstack([meta(r) for r in rids]))
TR=feats(train); TE=feats(test)
def mg(P):
    au=[roc_auc_score(Y[:,c],P[:,c]) for c in range(19) if 0<Y[:,c].sum()<len(Y)]
    return float(np.mean(au)), float(roc_auc_score(Y.ravel(),P.ravel()))
def cv(X,C):
    oof=np.zeros((len(train),19))
    for seed in (0,1,2):
        for tr,va in KFold(5,shuffle=True,random_state=seed).split(train):
            for c in range(19):
                y=Y[tr,c]
                if y.sum()==0: continue
                if y.sum()==len(y): oof[va,c]+=1; continue
                sc=StandardScaler().fit(X[tr]); m=LogisticRegression(C=C,max_iter=1000,class_weight='balanced').fit(sc.transform(X[tr]),y)
                oof[va,c]+=m.predict_proba(sc.transform(X[va]))[:,1]
    return oof/3.0
def fit_test(Xtr,Xte,C):
    P=np.zeros((len(test),19))
    for c in range(19):
        y=Y[:,c]
        if y.sum()==0: continue
        sc=StandardScaler().fit(Xtr); m=LogisticRegression(C=C,max_iter=1000,class_weight='balanced').fit(sc.transform(Xtr),y)
        P[:,c]=m.predict_proba(sc.transform(Xte))[:,1]
    return P
combos={'clap+meta':['clap','meta'],'clap':['clap'],
        'perch+clap+meta':['perch','perchmax','clap','meta'],
        'perch+clap+pmax+meta':['perch','perchmax','clap','pmax','meta']}
res={}
for name,keys in combos.items():
    X=np.hstack([TR[k] for k in keys])
    for C in (0.03,0.1,0.3):
        oof=cv(X,C); m,g=mg(oof); res[f'{name}|C{C}']=(m,g)
        print(f'{name:24s} C={C} macro={m:.4f} global={g:.4f}',flush=True)
# export clap+meta (diverse) and best combo
for name,keys,C in [('clap_meta',['clap','meta'],0.03),('perch_clap_pmax_meta',['perch','perchmax','clap','pmax','meta'],0.3)]:
    Xtr=np.hstack([TR[k] for k in keys]); Xte=np.hstack([TE[k] for k in keys])
    oof=cv(Xtr,C); m,g=mg(oof); P=fit_test(Xtr,Xte,C)
    fnm=f'oof_{name}'
    np.savez(f'{OUT}/{fnm}.npz',rec_ids=np.array(train),probs=oof,test_rec_ids=np.array(test),test_probs=P)
    json.dump({'name':name,'macro':m,'global':g,'C':C},open(f'{OUT}/{fnm}.json','w'),indent=2)
    print("EXPORTED",name,round(m,4),round(g,4),flush=True)
json.dump(res,open(f'{OUT}/clap_res.json','w'),indent=2)
