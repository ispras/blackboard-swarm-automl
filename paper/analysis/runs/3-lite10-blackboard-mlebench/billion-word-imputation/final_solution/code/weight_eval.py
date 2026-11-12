import pickle,sys,argparse
import numpy as np
from Levenshtein import distance as lev
COMMON=["the","a","an","to","of","and","in","that","is","for","on","it","was","with","as","at","by","be","this","from"]
def norm(w,p):
    s=sum(p) or 1; return {a:b/s for a,b in zip(w,p)}
def cands(ws):
    c=set(ws); c.update(COMMON)
    for w in ws:
        n=len(w)
        for L in range(1,n): c.add(w[:L]); c.add(w[L:])
        for i in range(n): c.add(w[:i]+w[i+1:])
    return list(c)
def choose(comb,level=1.0,lam=0.05):
    items=sorted(comb.items(),key=lambda kv:-kv[1])[:20]
    ws=[w for w,_ in items]; ps=np.array([v for _,v in items],float)
    lg=np.log(ps+1e-12)/level; ps=np.exp(lg-lg.max()); ps/=ps.sum()
    best=None;bw=None
    for c in cands(ws):
        e=float(sum(ps[j]*lev(c,ws[j]) for j in range(len(ws))))+lam*len(c)
        if best is None or e<best: best=e;bw=c
    return bw
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--w',required=True); ap.add_argument('--out',default='')
    ap.add_argument('--T',type=float,default=1.0); ap.add_argument('--lam',type=float,default=0.05)
    args=ap.parse_args(); W=[float(x) for x in args.w.split(',')]
    gaps=pickle.load(open('hold_chosen_gaps_mix3.pkl','rb')); conf=np.load('hold_conf_mix3.npy')
    Z=pickle.load(open('/download_job_code/payload/sandbox/workspaces/spica_psi/hold_mix3_zs.pkl','rb'))
    F=pickle.load(open('/download_job_code/payload/sandbox/workspaces/spica_psi/hold_mix3_ft.pkl','rb'))
    B=pickle.load(open('/download_job_code/payload/sandbox/workspaces/spica_psi/hold_mix3_base.pkl','rb'))
    toks=[l.rstrip('\n').split() for l in open('/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout/holdout_gap.txt')]
    orig=[l.rstrip('\n') for l in open('/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout/holdout_orig.txt')]
    N=len(toks); base=[lev(orig[i],' '.join(toks[i])) for i in range(N)]
    chosen=[None]*N
    for i in range(N):
        k=gaps[i]
        if not k or k<1 or k>len(toks[i])-1: continue
        comb={}
        for dump,wt in zip([Z,F,B],W):
            if wt<=0: continue
            g=dump[i].get(k) if i<len(dump) else None
            if not g: continue
            d=norm(g['w'][:20],g['p'][:20])
            for w,v in d.items(): comb[w]=comb.get(w,0)+wt*v
        if not comb: continue
        chosen[i]=choose(comb,args.T,args.lam)
    res={}
    for tau in [0.45,0.5,0.55,0.6]:
        tot=0
        for i in range(N):
            if chosen[i] is None or conf[i]<tau: tot+=base[i]; continue
            k=gaps[i]; tot+=lev(' '.join(toks[i][:k]+[chosen[i]]+toks[i][k:]),orig[i])
        res[tau]=tot/N
    best=min(res.items(),key=lambda kv:kv[1])
    print('W',W,'T',args.T,'lam',args.lam,'metrics',{k:round(v,4) for k,v in res.items()},'BEST',round(best[1],4),'tau',best[0])
    if args.out: pickle.dump(chosen,open(args.out,'wb'))
if __name__=='__main__': main()
