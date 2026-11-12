import json, csv, math, time, pickle, os, sys, zipfile, io, statistics
import numpy as np
import lightgbm as lgb
import Levenshtein

BASE='/download_job_code/payload/sandbox/workspaces/betelgeuse_tau'
FOM='/download_job_code/payload/sandbox/workspaces/fomalhaut_zeta'
VER='/download_job_code/payload/sandbox/workspaces/mirfak_chi'
PUNCT={',','.',';',':','!','?','"',"'",'(',')','--','-','``',"''"}
def norm(t): return t.replace('\u0120',' ').replace('\u010a',' ').replace('\u0109',' ').strip()

def load_jsonl_map(path, key='id'):
    d={}
    for l in open(path):
        o=json.loads(l); d[o[key]]=o
    return d

def load_extra(prefix):
    # returns dict id -> {gap:(raw,prob)} for verifier files
    d={}
    for l in open(prefix):
        o=json.loads(l); d[o['id']]={int(g):(float(r),float(p)) for g,r,p in o['gap_probs']}
    return d

def load_mass(prefix):
    d={}
    for l in open(prefix):
        o=json.loads(l); d[o['id']]={int(g):float(p) for g,p in o['gap_probs']}
    return d

def build_features(T, mlm_pred, mass, cs, vd, vd2, vd3):
    n=len(T)
    items=sorted(mass.items(), key=lambda kv:-kv[1])[:10]
    topd=dict(items); rest=max(0.0,1.0-sum(topd.values()))
    gaps_all=[]
    for gs,d in mlm_pred.items():
        g=int(gs)
        if 1<=g<n: gaps_all.append((g,d))
    if not gaps_all: return [],[]
    missing=[g for g,_ in gaps_all if g not in topd]
    floor=math.log(max(rest,1e-9)/max(1,len(missing))) if missing else -30.0
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
        def vfeat(vd):
            if not vd: return 0.0,0.0,0.0
            vr=[x[0] for x in vd.values()]; vmin=min(vr)
            raw=vd[g][0] if g in vd else vmin-2.0
            p=vd[g][1] if g in vd else 0.0
            mu2=sum(vr)/len(vr); sd2=(sum((x-mu2)**2 for x in vr)/len(vr))**0.5 or 1.0
            return raw,(raw-mu2)/sd2,p
        vraw,vz,vp=vfeat(vd)
        vraw2,vz2,vp2=vfeat(vd2)
        vraw3,vz3,vp3=vfeat(vd3)
        rows.append([fl,top1,top2,l20,top1-l20,top1-top2,top1-lp_nxt,top1-lp_prv,
                     g/n,(n-g)/n,len(T[g-1]),len(T[g]),
                     float(T[g-1] in PUNCT),float(T[g] in PUNCT),
                     float(g==1),float(g==n-1),1,
                     sg,zsg,rat,sgp,rank,vraw,vz,vp,vraw2,vz2,vp2,vraw3,vz3,vp3])
        gaps.append(g)
    return rows,gaps

def stage_val():
    truth=[json.loads(l) for l in open(f'{BASE}/val_truth.jsonl')]
    vt={int(r['id']):r['sentence'] for r in csv.DictReader(open(f'{BASE}/val_test.csv'))}
    tg={d['id']:d['gap'] for d in truth}; orig={d['id']:d['orig'] for d in truth}
    fom=load_mass(f'{FOM}/fomalhaut_val_gap_probs20m.jsonl')
    mlm={}
    for l in open(f'{BASE}/mlm_val20k.jsonl'):
        d=json.loads(l); mlm[d['id']]=d['pred']
    causal={}
    for l in open(f'{FOM}/causal_surp_val.jsonl'):
        d=json.loads(l); causal[d['id']]=d['surp']
    v1=load_extra(f'{VER}/verif_gaps_val.jsonl')
    v2=load_extra(f'{VER}/verif_ft_gaps_val.jsonl')
    v3=load_extra(f'{VER}/verif_large_bgt_val.jsonl')
    X=[];Y=[];rid=[];meta=[]
    for d in truth:
        i=d['id']; T=vt[i].split(' ')
        rows,gaps=build_features(T, mlm[i], fom[i], causal.get(i,[]), v1.get(i,{}), v2.get(i,{}), v3.get(i,{}))
        for r,g in zip(rows,gaps):
            X.append(r); Y.append(1 if g==tg[i] else 0); rid.append(i)
        meta.append((i,gaps))
    X=np.array(X,dtype=np.float32); Y=np.array(Y,dtype=np.int8); rid=np.array(rid)
    tr=rid<5000
    clf=lgb.LGBMClassifier(objective='binary',num_leaves=31,learning_rate=0.05,n_estimators=400,min_child_samples=50,n_jobs=4,verbose=-1)
    clf.fit(X[tr],Y[tr])
    s=clf.predict_proba(X)[:,1]
    sb={}; k=0
    for i,gaps in meta:
        sb[i]=s[k:k+len(gaps)]; k+=len(gaps)
    t1=t3=0; n=0; ranks=[]
    for i,gaps in meta:
        if i<5000: continue
        order=np.argsort(-sb[i]); rank=list(np.array(gaps)[order]).index(tg[i])
        t1+=rank==0; t3+=rank<3; n+=1; ranks.append(rank)
    print('V7 HOLDOUT top1 %.4f top3 %.4f meanrank %.3f n %d'%(t1/n,t3/n,np.mean(ranks),n),flush=True)
    # gated policy + medoid base+large+trigram v2
    def medoid(P,mpb=0.03,depth=3,ml=4.57):
        ws=list(P);pool=set(ws)
        for w in ws:
            for kk in range(1,depth+1):
                if len(w)>kk: pool.add(w[:kk])
        pool|={',','the','a','of','to','and','in','.'}
        best=None
        for x in pool:
            e=sum(P[w]*Levenshtein.distance(x,w) for w in ws)+mpb*(len(x)+ml)
            if best is None or e<best[0]: best=(e,x)
        return best[1]
    lmtrue={}
    for l in open(f'{BASE}/val_v4_lm.jsonl'):
        o=json.loads(l); lmtrue[o['id']]=(o['base'],o['large'])
    tri={}
    for l in open(f'{FOM}/trigram_val_chosen_v2.jsonl'):
        o=json.loads(l); tri[o['id']]=o
    def policy(thr):
        tot=0;n2=0
        for i,gaps in meta:
            if i<5000: continue
            n2+=1; T=vt[i].split(' '); noop=' '.join(T)
            sc=sb[i]; order=np.argsort(-sc); pg=gaps[order[0]]
            mg=sc[order[0]]-sc[order[1]] if len(order)>1 else 99
            if mg<thr:
                tot+=Levenshtein.distance(orig[i],noop); continue
            P={}
            if i in lmtrue:
                base,large=lmtrue[i]
                for w,lp in base[:20]: P[w]=P.get(w,0)+0.3*math.exp(lp)
                for w,lp in large[:20]: P[w]=P.get(w,0)+0.7*math.exp(lp)
            t=tri.get(i)
            if t and int(t.get('gap',-1))==pg:
                dd={}
                for w,c in (t.get('tri') or []):
                    if w and c>0: dd[w]=dd.get(w,0)+float(c)
                z=sum(dd.values()) or 1.0
                for w,v in dd.items(): P[w]=P.get(w,0)+0.4*v/z
            if not P:
                tot+=Levenshtein.distance(orig[i],noop); continue
            x=medoid(P)
            tot+=Levenshtein.distance(orig[i],' '.join(T[:pg]+[x]+T[pg:]))
        return tot/n2
    for th in [0.30,0.35,0.40,0.45]:
        print('V7 POLICY th %.2f -> %.4f'%(th,policy(th)),flush=True)
    # save model trained on ALL val for test application
    clf_all=lgb.LGBMClassifier(objective='binary',num_leaves=31,learning_rate=0.05,n_estimators=400,min_child_samples=50,n_jobs=4,verbose=-1)
    clf_all.fit(X,Y)
    pickle.dump({'model':clf_all},open(f'{VER}/v7_gbdt.pkl','wb'))
    print('saved model',flush=True)

if __name__=='__main__':
    stage_val()
