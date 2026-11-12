import pickle, numpy as np, argparse
from Levenshtein import distance as lev
HOLD='/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout'
COMMON=["the","a","an","to","of","and","in","that","is","for","on","it","was","with","as","at","by","be","this","from"]
def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--det', default='det_hold_mix.pkl')
    ap.add_argument('--zs', default='hold_mix_zs.pkl')
    ap.add_argument('--ft', default='hold_mix_ft.pkl')
    ap.add_argument('--base', default='hold_mix_base.pkl')
    ap.add_argument('--weights', default='0.5,0.3,0.2')
    ap.add_argument('--topk', type=int, default=20)
    args=ap.parse_args()
    det=pickle.load(open(args.det,'rb')); gp=det['gap_probs']; toks=det['tokens']
    zs=pickle.load(open(args.zs,'rb')); ft=pickle.load(open(args.ft,'rb')); base=pickle.load(open(args.base,'rb'))
    orig=[l.rstrip('\n') for l in open(HOLD+'/holdout_orig.txt')]
    gapf=[l.rstrip('\n') for l in open(HOLD+'/holdout_gap.txt')]
    meta=[l.rstrip('\n').split('\t') for l in open(HOLD+'/holdout_meta.tsv')][1:]
    W=np.array([float(x) for x in args.weights.split(',')]); Ws=W.sum()
    n=len(gp)
    S=[]
    for i in range(n):
        a=np.asarray(gp[i]); M=len(toks[i])
        if len(a)==0: S.append(None); continue
        j=int(np.argmax(a)); k=j+1; p=float(a[j])
        if k<1 or k>M-1: S.append(None); continue
        comb={}
        for dump,wt in zip([zs,ft,base],W):
            g=dump[i].get(k) if i<len(dump) else None
            if g is None: continue
            d={}
            for wq,pp in zip(g['w'][:args.topk],g['p'][:args.topk]): d[wq]=d.get(wq,0.0)+pp
            d=norm(d)
            for wq,v in d.items(): comb[wq]=comb.get(wq,0.0)+wt*v/Ws
        if not comb: S.append((k,p,None)); continue
        items=sorted(comb.items(), key=lambda kv:-kv[1])[:args.topk]
        ws=[w for w,_ in items]; ps=np.array([x for _,x in items],dtype=np.float64)
        cands=set(ws)
        for wq in ws:
            for L in range(1,len(wq)+1): cands.add(wq[:L])
        cands.update(COMMON); cands=list(cands)
        levL=np.zeros((len(cands),len(ws)),dtype=np.float64)
        for ci,c in enumerate(cands):
            for wj,wq in enumerate(ws): levL[ci,wj]=lev(c,wq)
        lens=np.array([len(c) for c in cands],dtype=np.float64)
        S.append((k,p,(ws,ps,cands,levL,lens)))
    base_costs=np.array([lev(gapf[i],orig[i]) for i in range(n)])
    print('n',n,'baseline',base_costs.mean())
    for tau in [0.45,0.5,0.55,0.6]:
        for level in [1.0,1.25,1.5]:
            for lam in [0.0,0.05,0.1]:
                tot=0.0;app=0;loc=0;we=0
                for i in range(n):
                    r=S[i]
                    if r is None or r[2] is None or r[1]<tau:
                        tot+=base_costs[i]; continue
                    k,p,(ws,ps,cands,levL,lens)=r
                    lg=np.log(ps+1e-12)/level; q=np.exp(lg-lg.max()); q/=q.sum()
                    E=levL@q+lam*lens
                    ci=int(np.argmin(E)); bw=cands[ci]
                    tokss=gapf[i].split()
                    pred=' '.join(tokss[:k]+[bw]+tokss[k:]); app+=1
                    rmi=int(meta[i][2]); rmw=meta[i][3]
                    if k==rmi: loc+=1
                    if k==rmi and bw==rmw: we+=1
                    tot+=lev(pred,orig[i])
                print(f'tau={tau} level={level} lam={lam}: metric {tot/n:.4f} apply {app/n:.3f} loc {loc/max(1,app):.3f} word {we/max(1,app):.3f}', flush=True)
if __name__=='__main__': main()
