"""Train fused gap ranker on shared val (fomalhaut 20M + MLM all-gap), fit policy,
save model+config for test inference. Test-simulated truncation: fomalhaut top-10."""
import json, csv, math, statistics, sys, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import Levenshtein

FOM='../fomalhaut_zeta/fomalhaut_val_gap_probs20m.jsonl'
FOMCANDS='../fomalhaut_zeta/fomalhaut_gap_cands_val.jsonl'
MLM='mlm_val20k.jsonl'
test={int(r['id']):r['sentence'] for r in csv.DictReader(open('val_test.csv',encoding='utf-8'))}
truth={json.loads(l)['id']:json.loads(l) for l in open('val_truth.jsonl',encoding='utf-8')}
fom={}
for l in open(FOM,encoding='utf-8'):
    o=json.loads(l); fom[int(o['id'])]=o
mlm={}
for l in open(MLM,encoding='utf-8'):
    o=json.loads(l); mlm[int(o['id'])]=o
fom_cands={}
for l in open(FOMCANDS,encoding='utf-8'):
    o=json.loads(l); fom_cands[int(o['id'])]=o
print('loaded',len(fom),len(mlm),len(fom_cands),flush=True)
PUNCT=set([',','.',';',':','!','?','"',"'",'(',')','--','-','``',"''"])
TOPN=10

def fom_mass(o):
    gp=o.get('gap_probs') or []
    if isinstance(gp,dict): gp=list(gp.items())
    return {int(g):float(p) for g,p in gp if float(p)>0}

def build(i):
    T=[t for t in test[i].split(' ') if t!='']
    n=len(T); mass=fom_mass(fom[i]); pred=mlm[i]['pred']
    items=sorted(mass.items(), key=lambda kv:-kv[1])[:TOPN]
    topd=dict(items); rest=1.0-sum(topd.values())
    gm=[]
    for gs,d in pred.items():
        g=int(gs)
        if not (1<=g<n): continue
        pairs=[(w,lp) for w,lp in zip(d['w'],d['lp']) if w][:15]
        if len(pairs)>=2: gm.append((g,pairs))
    if not gm: return None
    missing=[g for g,_ in gm if g not in topd]
    floor=math.log(max(rest,1e-9)/max(1,len(missing))) if missing else -30.0
    rows=[];gaps=[]
    for g,pairs in gm:
        fl=math.log(max(topd[g],1e-12)) if g in topd else floor
        ws=[w for w,_ in pairs]; lps=[lp for _,lp in pairs]; top1=lps[0]; mx=max(lps)
        nxt=T[g]; prv=T[g-1]
        lp_nxt=min(lps) if nxt not in ws else lps[ws.index(nxt)]
        lp_prv=min(lps) if prv not in ws else lps[ws.index(prv)]
        nconf=sum(1 for lp in lps if math.exp(lp)>0.01)
        punct=1.0 if (nxt in PUNCT or prv in PUNCT) else 0.0
        rows.append([fl,top1,mx-lp_nxt,mx-lp_prv,top1-lps[-1],top1-lps[1],
                     g/(n-1),nconf,len(ws[0]),punct])
        gaps.append(g)
    A=np.array(rows);mu=A.mean(0);sd=A.std(0);sd[sd<1e-9]=1.0;Z=(A-mu)/sd
    inter=np.column_stack([Z[:,0]*Z[:,2],Z[:,0]*Z[:,3],Z[:,2]*Z[:,4]])
    Z=np.column_stack([Z,inter])
    return T,gaps,Z,pred
ids=sorted(set(fom)&set(mlm)&set(truth))
cache={}
for i in ids:
    b=build(i)
    if b: cache[i]=b
print('cached',len(cache),flush=True)
def mk(idset):
    X=[];y=[];meta=[]
    for i in idset:
        T,gaps,Z,pred=cache[i];tg=truth[i]['gap']
        for j,g in enumerate(gaps):X.append(Z[j]);y.append(1 if g==tg else 0)
        meta.append((i,gaps,Z))
    return np.array(X),np.array(y),meta
tr=[i for i in ids if i<6000 and i in cache];te=[i for i in ids if i>=6000 and i in cache]
Xtr,ytr,_=mk(tr);Xte,yte,meta_te=mk(te)
print('train',Xtr.shape,'test',Xte.shape,flush=True)
clf=LogisticRegression(C=0.1,max_iter=3000).fit(Xtr,ytr)
dat={};top1=0;top3=0
for i,gaps,Z in meta_te:
    s=clf.decision_function(Z);e=np.exp(s-s.max());P=e/e.sum()
    order=np.argsort(-P);tg=truth[i]['gap']
    rank=list(np.array(gaps)[order]).index(tg) if tg in gaps else 99
    top1+=rank==0;top3+=rank<3
    dat[i]={int(g):float(pp) for g,pp in zip(gaps,P)}
n=len(meta_te)
print(f'FUSED(trunc top10) held-out top1={top1/n:.4f} top3={top3/n:.4f} n={n}',flush=True)

def norm_cands(i,g):
    o=fom_cands.get(i); d={}
    if o:
        for it in (o.get('cands') or []):
            gg,w,c=int(it[0]),str(it[1]),float(it[2])
            if gg==g and w and c>0: d[w]=d.get(w,0)+c
    z=sum(d.values()) or 1.0
    return {w:v/z for w,v in d.items()}
def medoid(P, mpb=0.2, mean_len=4.57):
    ws=list(P);pool=set(ws)
    for w in ws:
        for k in (1,2,3):
            if len(w)>k:pool.add(w[:k])
    pool|={',','the','a','of','to','and','in','.'}
    best=None
    for x in pool:
        e=sum(P[w]*Levenshtein.distance(x,w) for w in ws)+mpb*(len(x)+mean_len)
        if best is None or e<best[0]:best=(e,x)
    return best[1]
def eval_policy(idset,alpha,mpb,th):
    ds=[];act=0
    for i in idset:
        T,gaps,Z,pred=cache[i];P=dat[i]
        g1=max(P,key=P.get);P1=P[g1]
        if P1<th:
            ds.append(Levenshtein.distance(truth[i]['orig'],' '.join(T)));continue
        d=pred.get(str(g1)) or {}
        Pw={}
        for w,lp in zip(d.get('w',[]),d.get('lp',[])):
            if w:Pw[w]=Pw.get(w,0)+math.exp(lp)
        fw=norm_cands(i,g1)
        for w,v in fw.items():Pw[w]=Pw.get(w,0)+alpha*v
        if not Pw:
            ds.append(Levenshtein.distance(truth[i]['orig'],' '.join(T)));continue
        x=medoid(Pw,mpb);act+=1
        ds.append(Levenshtein.distance(truth[i]['orig'],' '.join(T[:g1]+[x]+T[g1:])))
    return statistics.mean(ds), act/len(idset)
te_ids=[i for i,_,_ in meta_te]; sub=te_ids[:5000]
best=None
for alpha in [0.0,0.4]:
    for mpb in [0.1,0.2]:
        for th in [0.55,0.6,0.65]:
            m,cov=eval_policy(sub,alpha,mpb,th)
            print(f'policy a={alpha} mpb={mpb} th={th}: {m:.4f} cov={cov:.3f}',flush=True)
            if best is None or m<best[0]:best=(m,alpha,mpb,th)
# full held-out with best
mfull,covfull=eval_policy(te_ids,best[1],best[2],best[3])
print('BEST',best,'heldout_full',mfull,'cov',covfull,flush=True)
# retrain on ALL val
Xall=[];yall=[]
for i in ids:
    if i not in cache:continue
    T,gaps,Z,pred=cache[i];tg=truth[i]['gap']
    for j,g in enumerate(gaps):Xall.append(Z[j]);yall.append(1 if g==tg else 0)
clf_all=LogisticRegression(C=0.1,max_iter=3000).fit(np.array(Xall),np.array(yall))
cfg={'alpha':best[1],'mpb':best[2],'threshold':best[3],'topN':TOPN,
     'n_features':int(np.array(Xall).shape[1]),
     'val_heldout_top1':top1/n,'val_heldout_top3':top3/n,'val_heldout_policy':mfull,
     'val_heldout_cov':covfull,
     'external_models':[
       {'id':'roberta-base','source':'huggingface.co/roberta-base','license':'MIT','purpose':'MLM conditional word distribution per gap'},
       {'id':'GPT-2 medium causal (mirfak dev only)','source':'not used in final','purpose':'n/a'}
     ]}
pickle.dump(clf_all,open('fusion_model.pkl','wb'))
json.dump(cfg,open('fusion_config.json','w'),indent=2)
print('saved',cfg,flush=True)
