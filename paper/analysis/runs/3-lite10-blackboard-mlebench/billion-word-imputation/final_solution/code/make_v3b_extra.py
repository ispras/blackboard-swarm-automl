import pickle,argparse
import numpy as np
from Levenshtein import distance as lev
COMMON=["the","a","an","to","of","and","in","that","is","for","on","it","was","with","as","at","by","be","this","from"]
def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}
def make_cands(ws):
    c=set(ws); c.update(COMMON)
    for w in ws:
        n=len(w)
        for L in range(1,n): c.add(w[:L]); c.add(w[L:])
        for i in range(n): c.add(w[:i]+w[i+1:])
    return c
def choose(comb,level=1.0,lam=0.05,topk=20):
    items=sorted(comb.items(),key=lambda kv:-kv[1])[:topk]
    ws=[w for w,_ in items]; ps=np.array([x for _,x in items],dtype=float)
    lg=np.log(ps+1e-12)/level; ps=np.exp(lg-lg.max()); ps/=ps.sum()
    best=None;bw=None
    for c in make_cands(ws):
        e=float(sum(ps[j]*lev(c,ws[j]) for j in range(len(ws))))+lam*len(c)
        if best is None or e<best: best=e;bw=c
    cw=float(sum(ps[j]*lev(bw,ws[j]) for j in range(len(ws))))
    return bw,cw
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gaps',default='test_chosen_gaps_mix3.pkl'); ap.add_argument('--conf',default='test_conf_mix3.npy')
    ap.add_argument('--zs',default='/download_job_code/payload/sandbox/workspaces/spica_psi/test_v3_zs.pkl')
    ap.add_argument('--ft',default='/download_job_code/payload/sandbox/workspaces/spica_psi/test_v3_ft.pkl')
    ap.add_argument('--base',default='/download_job_code/payload/sandbox/workspaces/spica_psi/test_v3_base.pkl')
    ap.add_argument('--lo',type=float,default=0.30); ap.add_argument('--hi',type=float,default=0.50)
    ap.add_argument('--cwmax',type=float,default=0.30)
    ap.add_argument('--out',default='decisions_v3b_extra.tsv')
    args=ap.parse_args()
    gaps=pickle.load(open(args.gaps,'rb')); conf=np.load(args.conf)
    Z=pickle.load(open(args.zs,'rb')); F=pickle.load(open(args.ft,'rb')); B=pickle.load(open(args.base,'rb'))
    W=[0.5,0.3,0.2]; Ws=sum(W)
    n=len(conf); out=[]; cand=0
    for i in range(n):
        c=float(conf[i])
        if not (args.lo<=c<args.hi): continue
        k=gaps[i]
        if not k or k<1: continue
        comb={}
        for dump,wt in zip([Z,F,B],W):
            g=dump[i].get(k) if i<len(dump) else None
            if g is None: continue
            d=norm({w:pp for w,pp in zip(g['w'][:20],g['p'][:20])})
            for w,v in d.items(): comb[w]=comb.get(w,0.0)+wt*v/Ws
        if not comb: continue
        cand+=1
        bw,cw=choose(comb)
        if cw<=args.cwmax: out.append((i,k,bw,cw,c))
        if i%20000==0: print('i',i,'cand',cand,'out',len(out),flush=True)
    with open(args.out,'w') as f:
        for (i,k,bw,cw,c) in out: f.write(f'{i}\t{k}\t{bw}\n')
    print('candidate low-conf sentences',cand,'extra insertions',len(out),'->',args.out)
    print('sample',out[:5])
if __name__=='__main__': main()
