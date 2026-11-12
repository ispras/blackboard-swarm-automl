import json, csv, math, time, statistics, os, pickle
import numpy as np
import lightgbm as lgb
import Levenshtein
from collections import defaultdict

BASE = '/download_job_code/payload/sandbox/workspaces/betelgeuse_tau'
OUT = '/download_job_code/payload/sandbox/workspaces/fomalhaut_zeta'
PUNCT = {',','.',';',':','!','?','"',"'",'(',')','--','-','``',"''"}
def norm(t): return t.replace('\u0120',' ').replace('\u010a',' ').replace('\u0109',' ').strip()

t0=time.time()
truth=[json.loads(l) for l in open(f'{BASE}/val_truth.jsonl')]
vt={int(r['id']):r['sentence'] for r in csv.DictReader(open(f'{BASE}/val_test.csv'))}
tg={d['id']:d['gap'] for d in truth}; orig={d['id']:d['orig'] for d in truth}
fom={}
for l in open(f'{OUT}/fomalhaut_val_gap_probs20m.jsonl'):
    d=json.loads(l); fom[d['id']]={int(g):float(p) for g,p in d['gap_probs']}
mlm={}
for l in open(f'{BASE}/mlm_val20k.jsonl'):
    d=json.loads(l); mlm[d['id']]=d['pred']
causal={}
for l in open(f'{OUT}/causal_surp_val.jsonl'):
    d=json.loads(l); causal[d['id']]=d['surp']
verif={}
for l in open('../mirfak_chi/verif_gaps_val.jsonl'):
    d=json.loads(l); verif[d['id']]={int(g):(float(r),float(p)) for g,r,p in d['gap_probs']}
verif2={}
for l in open('../mirfak_chi/verif_ft_gaps_val.jsonl'):
    d=json.loads(l); verif2[d['id']]={int(g):(float(r),float(p)) for g,r,p in d['gap_probs']}
verif3={}
for l in open(f'{OUT}/verif_large_gaps_val.jsonl'):
    d=json.loads(l); verif3[d['id']]={int(g):(float(r),float(p)) for g,r,p in d['gap_probs']}
print('loaded', flush=True)

def fom_top10(mass):
    items=sorted(mass.items(), key=lambda kv:-kv[1])[:10]
    topd=dict(items); rest=max(0.0,1.0-sum(topd.values()))
    return topd, rest

def build_features(T, mlm_pred, mass, cs, vd, vd2, vd3):
    n=len(T); topd,rest=fom_top10(mass)
    gaps_all=[]
    for gs,d in mlm_pred.items():
        g=int(gs)
        if 1<=g<n: gaps_all.append((g,d))
    missing=[g for g,_ in gaps_all if g not in topd]
    floor=math.log(max(rest,1e-9)/max(1,len(missing))) if missing else -30.0
    # causal sentence stats
    if cs:
        arr=cs[1:] if len(cs)>1 else cs
        mu=sum(arr)/len(arr); sd=(sum((x-mu)**2 for x in arr)/len(arr))**0.5 or 1.0
        med=sorted(arr)[len(arr)//2] if arr else 0.0
    rows=[]; gaps=[]
    for g,d in gaps_all:
        if isinstance(d,dict):
            pairs=[(norm(w),lp) for w,lp in zip(d['w'],d['lp']) if norm(w)]
        else:
            pairs=[(norm(w),lp) for w,lp in d if norm(w)]
        pairs=pairs[:20]
        if len(pairs)<2: continue
        ws=[w for w,_ in pairs]; lps=[lp for _,lp in pairs]
        top1=lps[0];top2=lps[1];l20=lps[-1]
        nxt=T[g];prv=T[g-1]
        lp_nxt=min(lps) if nxt not in ws else lps[ws.index(nxt)]
        lp_prv=min(lps) if prv not in ws else lps[ws.index(prv)]
        fl=math.log(max(topd[g],1e-12)) if g in topd else floor
        if cs:
            sg=cs[g] if g<len(cs) else 0.0; zsg=(sg-mu)/sd; rat=sg/(med+1e-6)
            sgp=cs[g-1] if g-1>=0 else 0.0
            rank=sum(1 for x in arr if x<sg)/max(1,len(arr))
        else: sg=zsg=rat=sgp=rank=0.0
        if vd:
            vr_all=[x[0] for x in vd.values()]; vmin=min(vr_all)
            vraw=vd[g][0] if g in vd else vmin-2.0
            vp=vd[g][1] if g in vd else 0.0
            vmu=sum(vr_all)/len(vr_all); vsd=(sum((x-vmu)**2 for x in vr_all)/len(vr_all))**0.5 or 1.0
            vz=(vraw-vmu)/vsd
        else: vraw=vp=vz=0.0
        if vd2:
            vr2=[x[0] for x in vd2.values()]; vmin2=min(vr2)
            vraw2=vd2[g][0] if g in vd2 else vmin2-2.0
            vp2=vd2[g][1] if g in vd2 else 0.0
            vmu2=sum(vr2)/len(vr2); vsd2=(sum((x-vmu2)**2 for x in vr2)/len(vr2))**0.5 or 1.0
            vz2=(vraw2-vmu2)/vsd2
        else: vraw2=vp2=vz2=0.0
        if vd3:
            vr3=[x[0] for x in vd3.values()]; vmin3=min(vr3)
            vraw3=vd3[g][0] if g in vd3 else vmin3-2.0
            vp3=vd3[g][1] if g in vd3 else 0.0
            vmu3=sum(vr3)/len(vr3); vsd3=(sum((x-vmu3)**2 for x in vr3)/len(vr3))**0.5 or 1.0
            vz3=(vraw3-vmu3)/vsd3
        else: vraw3=vp3=vz3=0.0
        rows.append([fl,top1,top2,l20,top1-l20,top1-top2,top1-lp_nxt,top1-lp_prv,
                     g/n,(n-g)/n,len(T[g-1]),len(T[g]),
                     float(T[g-1] in PUNCT),float(T[g] in PUNCT),
                     float(g==1),float(g==n-1),1,
                     sg,zsg,rat,sgp,rank,vraw,vz,vp,vraw2,vz2,vp2,vraw3,vz3,vp3])
        gaps.append(g)
    return rows,gaps

# build val
X=[];Y=[];meta=[];rid=[]
for d in truth:
    i=d['id']; T=vt[i].split(' ')
    rows,gaps=build_features(T, mlm[i], fom[i], causal.get(i,[]), verif.get(i,{}), verif2.get(i,{}), verif3.get(i,{}))
    for r,g in zip(rows,gaps):
        X.append(r); Y.append(1 if g==tg[i] else 0); rid.append(i)
    meta.append((i,gaps,rows))
X=np.array(X,dtype=np.float32); Y=np.array(Y,dtype=np.int8)
print('val X',X.shape,'t=%.1f'%(time.time()-t0), flush=True)

def rank_eval(scores_by_id):
    ids=[i for i,_,_ in meta]
    # scores_by_id: dict id->np array aligned gaps
    top1=top3=0;n=0
    for i,gaps,_ in meta:
        s=scores_by_id[i]; order=np.argsort(-np.asarray(s)); rank=list(np.array(gaps)[order]).index(tg[i]); top1+=rank==0; top3+=rank<3; n+=1
    return top1/n,top3/n

# train on ids<5000, eval >=5000
tr=np.array(rid)<5000
clf=lgb.LGBMClassifier(objective='binary',num_leaves=31,learning_rate=0.05,n_estimators=300,min_child_samples=50,n_jobs=4,verbose=-1)
clf.fit(X[tr],Y[tr])
s=clf.predict_proba(X)[:,1]
sb={}
k=0
for i,gaps,rows in meta:
    sb[i]=s[k:k+len(gaps)]; k+=len(gaps)
ev_ids=[i for i in sb if i>=5000]
def ranked_subset(idset):
    t1=t3=0
    for i in idset:
        gaps=[g for (ii,gaps,_) in [(i,)+tuple() ] ] if False else None
    return None
# eval
t1=t3=0
for i,gaps,_ in meta:
    if i<5000: continue
    order=np.argsort(-sb[i]); rank=list(np.array(gaps)[order]).index(tg[i]); t1+=rank==0; t3+=rank<3
n=sum(1 for i,_,_ in meta if i>=5000)
print('HOLDOUT GBDT causal top1 %.4f top3 %.4f'%(t1/n,t3/n), flush=True)

# policy eval with base mlm words + medoid
def medoid(pairs):
    ws=[w for w,_ in pairs]; pr=[math.exp(lp) for _,lp in pairs]; tot=sum(pr); unk=max(0,1-tot)
    pool=set(ws)|{',','the','a','of','to','and','in','.'}
    for w in ws:
        for kk in (1,2):
            if len(w)>kk: pool.add(w[:kk])
    best=None
    for x in pool:
        e=sum(p*Levenshtein.distance(x,w) for w,p in zip(ws,pr))+unk*(len(x)+4.57)
        if best is None or e<best[0]: best=(e,x)
    return best[1]
def policy_on(idset, thr):
    ds=[]
    for i,gaps,rows in meta:
        if i not in idset: continue
        s=sb[i]; order=np.argsort(-s); pg=gaps[order[0]]
        mg=s[order[0]]-s[order[1]] if len(order)>1 else 99
        T=vt[i].split(' '); noop=' '.join(T)
        if mg<thr:
            ds.append(Levenshtein.distance(orig[i],noop)); continue
        d=mlm[i].get(str(pg))
        pairs=[(norm(w),lp) for w,lp in zip(d['w'],d['lp']) if norm(w)] if d else []
        if len(pairs)<2:
            ds.append(Levenshtein.distance(orig[i],noop)); continue
        w=medoid(pairs)
        ds.append(Levenshtein.distance(orig[i],' '.join(T[:pg]+[w]+T[pg:])))
    return statistics.mean(ds)
trset=set(i for i in sb if i<5000); evset=set(i for i in sb if i>=5000)
best=min(((policy_on(trset,th),th) for th in [0.02,0.05,0.08,0.1,0.12,0.15,0.2,0.25,0.3,0.4]))
print('train-tuned thr',best,flush=True)
for th in [0.25,0.3,0.35,0.4,0.45,0.5]:
    print('HOLDOUT policy th %.2f -> %.4f'%(th,policy_on(evset,th)),flush=True)

# train final on all val
clf_all=lgb.LGBMClassifier(objective='binary',num_leaves=31,learning_rate=0.05,n_estimators=300,min_child_samples=50,n_jobs=4,verbose=-1)
clf_all.fit(X,Y)
pickle.dump({'model':clf_all,'threshold':best[1]}, open(f'{OUT}/final_gbdt_v5.pkl','wb'))
json.dump({'thr_train':best[1],'holdout_top1':t1/n,'holdout_top3':t3/n,
           'policy_train_thr':policy_on(evset,best[1]),'policy_020':policy_on(evset,0.20)},
          open(f'{OUT}/final_val_metrics_v5.json','w'))
print('saved final model', flush=True)
