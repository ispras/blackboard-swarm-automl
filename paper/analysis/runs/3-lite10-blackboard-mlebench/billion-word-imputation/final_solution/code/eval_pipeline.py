import pickle, numpy as np, argparse
from Levenshtein import distance as lev
HOLD='/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout'

def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}

def word_dist(g, k=20):
    ws=g['w'][:k]; ps=g['p'][:k]
    d={}
    for w,p in zip(ws,ps): d[w]=d.get(w,0.0)+p
    return norm(d)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--det', default='/download_job_code/payload/sandbox/workspaces/arcturus_kappa/det_hold30k_ck11000.pkl')
    ap.add_argument('--zs', default='hold_gaps5_zs.pkl')
    ap.add_argument('--ft', default='hold_gaps5_ft.pkl')
    ap.add_argument('--base', default='hold_gaps5_base.pkl')
    ap.add_argument('--weights', default='0.5,0.3,0.2')
    ap.add_argument('--temp', type=float, default=1.5)
    ap.add_argument('--lam', type=float, default=0.05)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--n', type=int, default=0)
    args=ap.parse_args()
    det=pickle.load(open(args.det,'rb'))
    gp=det['gap_probs']; toks=det['tokens']
    zs=pickle.load(open(args.zs,'rb')); ft=pickle.load(open(args.ft,'rb')); base=pickle.load(open(args.base,'rb'))
    orig=[l.rstrip('\n') for l in open(HOLD+'/holdout_orig.txt')]
    gapf=[l.rstrip('\n') for l in open(HOLD+'/holdout_gap.txt')]
    meta=[l.rstrip('\n').split('\t') for l in open(HOLD+'/holdout_meta.tsv')][1:]
    W=np.array([float(x) for x in args.weights.split(',')])
    n=len(gp)
    if args.n: n=min(n,args.n)
    # precompute chosen gap, conf, word for each
    recs=[]
    for i in range(n):
        a=np.asarray(gp[i]); M=len(toks[i])
        if len(a)==0: recs.append(None); continue
        j=int(np.argmax(a)); k=j+1; conf=float(a[j])
        if k<1 or k>M-1: recs.append(None); continue
        # combine word dists at gap k
        comb={}
        for dump,wt in zip([zs,ft,base],W):
            if wt<=0: continue
            g=dump[i].get(k) if i<len(dump) else None
            if g is None: continue
            d=word_dist(g,args.topk)
            for w,v in d.items(): comb[w]=comb.get(w,0.0)+wt*v/W.sum()
        if not comb: recs.append((k,conf,None)); continue
        items=sorted(comb.items(), key=lambda kv:-kv[1])[:args.topk]
        ws=[w for w,_ in items]; ps=np.array([p for _,p in items])
        if args.temp!=1.0:
            lg=np.log(ps+1e-12)/args.temp; ps=np.exp(lg-lg.max())
        ps/=ps.sum()
        best=None
        for c in ws:
            e=float(sum(ps[jj]*lev(c,ws[jj]) for jj in range(len(ws))))+args.lam*len(c)
            if best is None or e<best[0]: best=(e,c)
        recs.append((k,conf,best[1]))
    base_costs=[lev(gapf[i],orig[i]) for i in range(n)]
    print('n',n,'baseline',np.mean(base_costs))
    for tau in [0.0,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85]:
        tot=0.0; app=0; loc=0; we=0
        for i in range(n):
            r=recs[i]
            if r is None or r[2] is None or r[1]<tau:
                tot+=lev(gapf[i],orig[i]); continue
            k,conf,w=r
            tokss=gapf[i].split()
            pred=' '.join(tokss[:k]+[w]+tokss[k:])
            tot+=lev(pred,orig[i]); app+=1
            rmi=int(meta[i][2]); rmw=meta[i][3]
            if k==rmi: loc+=1
            if k==rmi and w==rmw: we+=1
        print(f'tau {tau}: metric {tot/n:.4f} apply {app/n:.3f} locprec {loc/max(1,app):.3f} wordacc {we/max(1,app):.3f}')
if __name__=='__main__': main()
