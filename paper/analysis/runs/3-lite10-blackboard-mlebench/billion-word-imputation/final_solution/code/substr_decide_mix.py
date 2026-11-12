import pickle, numpy as np, csv, sys
from Levenshtein import distance as lev
SP='/download_job_code/payload/sandbox/workspaces/spica_psi/'
TEST='/download_job_code/payload/sandbox/workspaces/antares_upsilon/test_v2.txt'
zs=pickle.load(open(SP+'test_v2_zs.pkl','rb'))
ft=pickle.load(open(SP+'test_v2_ft.pkl','rb'))
base=pickle.load(open(SP+'test_v2_base.pkl','rb'))
G=pickle.load(open(SP+'test_chosen_gaps_mix.pkl','rb'))
C=np.load(SP+'test_chosen_conf_mix.npy')
W=[0.5,0.3,0.2]; Ws=sum(W); topk=20; NSUB=5; level=1.25; lam=0.05; tau=0.50
COMMON=["the","a","an","to","of","and","in","that","is","for","on","it","was","with","as","at","by","be","this","from"]
def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}
def choose(i,k):
    comb={}
    for dump,wt in zip([zs,ft,base],W):
        g=dump[i].get(k)
        if g is None: continue
        d={}
        for w,p in zip(g['w'][:topk],g['p'][:topk]): d[w]=d.get(w,0.0)+p
        d=norm(d)
        for w,v in d.items(): comb[w]=comb.get(w,0.0)+wt*v/Ws
    if not comb: return None
    items=sorted(comb.items(), key=lambda kv:-kv[1])[:topk]
    ws=[w for w,_ in items]; ps=np.array([x for _,x in items])
    lg=np.log(ps+1e-12)/level; ps=np.exp(lg-lg.max()); ps/=ps.sum()
    cands=set(ws)
    for w in ws[:NSUB]:
        if len(w)<=20:
            for a in range(len(w)):
                for b in range(a+1,len(w)+1): cands.add(w[a:b])
        else:
            for L in range(1,len(w)+1): cands.add(w[:L])
    cands.update(COMMON)
    best=None; bw=None
    for c in cands:
        e=sum(ps[j]*lev(c,ws[j]) for j in range(len(ws)))+lam*len(c)
        if best is None or e<best: best=e; bw=c
    return bw
ntok=[]
with open(TEST,newline='',encoding='utf-8') as f:
    r=csv.reader(f); next(r)
    for row in r: ntok.append(len(row[1].split(' ')))
out=open('ngram/decisions_mix_substr.tsv','w')
napp=0; skip=0
for i in range(len(G)):
    g=G[i]
    if not g: skip+=1; continue
    k=int(g[0]); c=float(C[i])
    if c<tau: continue
    if k<1 or k>ntok[i]-1: skip+=1; continue
    w=choose(i,k)
    if not w: skip+=1; continue
    out.write(f'{i}\t{k}\t{w}\n'); napp+=1
out.close()
print('applied',napp,'skipped',skip,'of',len(G))
