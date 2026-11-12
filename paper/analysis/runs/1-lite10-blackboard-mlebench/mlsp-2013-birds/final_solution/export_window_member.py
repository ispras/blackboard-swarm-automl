import os, sys, json, math, datetime
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
    doy=datetime.date(y,mo,d).timetuple().tm_yday
    mins=int(t[:2])*60+int(t[2:4])+int(t[4:6])/60
    v=[1.0 if i==si[st] else 0.0 for i in range(len(stations))]
    v+=[1.0 if i==di[date] else 0.0 for i in range(len(dates))]
    v+=[1.0 if y==2010 else 0.0]
    v+=[math.sin(2*math.pi*mins/1440),math.cos(2*math.pi*mins/1440),mins/1440.0]
    return np.array(v,dtype=np.float32)

z=np.load(f'{OUT}/ast_window_feats.npz',allow_pickle=True)
recs=list(z['recs']); idx={r:i for i,r in enumerate(recs)}
def g(name, ids): return np.vstack([z[name][idx[r]] for r in ids]).astype(np.float32)

# optional rigel AST embeddings
ast_ok=os.path.exists(f'{RIGEL}/ast_feats.npz')
if ast_ok:
    az=np.load(f'{RIGEL}/ast_feats.npz',allow_pickle=True)
    arecs=list(az['recs']); aidx={r:i for i,r in enumerate(arecs)}
    def gast(ids): return np.vstack([az['emb'][aidx[r]] for r in ids]).astype(np.float32)

def build(ids, keys, use_meta=True, use_ast=False):
    parts=[g(k,ids) for k in keys]
    if use_ast: parts.append(gast(ids))
    if use_meta: parts.append(np.vstack([meta(r) for r in ids]))
    return np.hstack(parts)

def mg(yp):
    au=[roc_auc_score(Y[:,c],yp[:,c]) for c in range(19) if 0<Y[:,c].sum()<len(Y)]
    return float(np.mean(au)), float(roc_auc_score(Y.ravel(),yp.ravel()))

def cv(X, C=0.1):
    oof=np.zeros((len(train),19))
    for seed in (0,1,2):
        for tr,va in KFold(5,shuffle=True,random_state=seed).split(train):
            for c in range(19):
                y=Y[tr,c]
                if y.sum()==0: continue
                if y.sum()==len(y): oof[va,c]+=1; continue
                sc=StandardScaler().fit(X[tr])
                m=LogisticRegression(C=C,max_iter=5000,class_weight='balanced').fit(sc.transform(X[tr]),y)
                oof[va,c]+=m.predict_proba(sc.transform(X[va]))[:,1]
    oof/=3.0
    return oof

def fit_test(Xtr,Xte,C=0.1):
    P=np.zeros((len(test),19))
    for c in range(19):
        y=Y[:,c]
        if y.sum()==0: continue
        sc=StandardScaler().fit(Xtr)
        m=LogisticRegression(C=C,max_iter=5000,class_weight='balanced').fit(sc.transform(Xtr),y)
        P[:,c]=m.predict_proba(sc.transform(Xte))[:,1]
    return P

def save_member(name, keys, use_meta=True, use_ast=False, C=0.1):
    Xtr=build(train,keys,use_meta,use_ast); Xte=build(test,keys,use_meta,use_ast)
    oof=cv(Xtr,C)
    mm,gg=mg(oof)
    tp=fit_test(Xtr,Xte,C)
    np.savez(f'{OUT}/oof_{name}.npz', rec_ids=np.array(train), probs=oof,
             test_rec_ids=np.array(test), test_probs=tp)
    json.dump({'name':name,'macro':mm,'global':gg,'C':C,'keys':keys,'meta':use_meta,'ast':use_ast},
              open(f'{OUT}/oof_{name}.json','w'), indent=2)
    print(f'{name:28s} macro={mm:.4f} global={gg:.4f}', flush=True)
    return gg

if __name__=='__main__':
    save_member('patch_max_meta', ['patch_max'], True, False, 0.1)
    save_member('patch_max_patchmean_meta', ['patch_max','patch_mean'], True, False, 0.1)
    if ast_ok:
        save_member('ast_patch_max_meta', ['patch_max'], True, True, 0.1)
        save_member('ast_patch_max', ['patch_max'], False, True, 0.1)
