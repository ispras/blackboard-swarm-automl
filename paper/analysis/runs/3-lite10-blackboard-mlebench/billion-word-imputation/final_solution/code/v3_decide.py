import pickle, numpy as np, csv
from Levenshtein import distance as lev
arc='/download_job_code/payload/sandbox/workspaces/arcturus_kappa/'
sp='/download_job_code/payload/sandbox/workspaces/spica_psi/'
TEST='/download_job_code/payload/sandbox/workspaces/antares_upsilon/test_v2.txt'
zs=pickle.load(open(sp+'test_v3_zs.pkl','rb')); ft=pickle.load(open(sp+'test_v3_ft.pkl','rb')); base=pickle.load(open(sp+'test_v3_base.pkl','rb'))
G=pickle.load(open(arc+'test_chosen_gaps_mix3.pkl','rb')); C=np.load(arc+'test_conf_mix3.npy')
W=[0.5,0.3,0.2]; Ws=sum(W); topk=20; level=1.0; lam=0.05; tau=0.50
COMMON=["the","a","an","to","of","and","in","that","is","for","on","it","was","with","as","at","by","be","this","from"]
def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}
def cands(ws):
    c=set(ws); c.update(COMMON)
    for w in ws:
        n=len(w)
        for L in range(1,n): c.add(w[:L]); c.add(w[L:])
        for j in range(n): c.add(w[:j]+w[j+1:])
    for w in ws[:5]:
        if len(w)<=20:
            for a in range(len(w)):
                for b in range(a+1,len(w)+1): c.add(w[a:b])
    return c
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
    best=None;bw=None
    for c in cands(ws):
        e=sum(ps[j]*lev(c,ws[j]) for j in range(len(ws)))+lam*len(c)
        if best is None or e<best: best=e;bw=c
    return bw
ntok=[]
with open(TEST,newline='',encoding='utf-8') as f:
    r=csv.reader(f); next(r)
    for row in r: ntok.append(len(row[1].split(' ')))
out=open('/download_job_code/payload/sandbox/workspaces/antares_upsilon/ngram/decisions_v3_substr.tsv','w')
napp=0; skip=0
for i in range(len(G)):
    g=G[i]
    if not g: skip+=1; continue
    k=int(g[0]) if isinstance(g,(list,tuple)) else int(g)
    if C[i]<tau: continue
    if k<1 or k>ntok[i]-1: skip+=1; continue
    w=choose(i,k)
    if not w: skip+=1; continue
    out.write(f'{i}\t{k}\t{w}\n'); napp+=1
out.close()
print('applied',napp,'skipped',skip,'of',len(G))
