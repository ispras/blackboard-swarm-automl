import pickle, numpy as np, argparse
from Levenshtein import distance as lev
HOLD='/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout'

def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}
def wd(g,k=20):
    d={}
    for w,p in zip(g['w'][:k],g['p'][:k]): d[w]=d.get(w,0.0)+p
    return norm(d)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--det', default='/download_job_code/payload/sandbox/workspaces/arcturus_kappa/det_hold30k_ck15625.pkl')
    ap.add_argument('--zs', default='hold_ch_zs_625.pkl')
    ap.add_argument('--ft', default='hold_ch_ft_625.pkl')
    ap.add_argument('--base', default='hold_ch_base_625.pkl')
    ap.add_argument('--weights', default='0.5,0.3,0.2')
    ap.add_argument('--temp', type=float, default=1.5)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--B', type=float, default=5.55)
    ap.add_argument('--risk', type=float, default=1.0, help='multiplier on (1-p) wrong-gap cost')
    ap.add_argument('--pc', type=float, default=1.0, help='exponent on p')
    args=ap.parse_args()
    det=pickle.load(open(args.det,'rb')); gp=det['gap_probs']; toks=det['tokens']
    zs=pickle.load(open(args.zs,'rb')); ft=pickle.load(open(args.ft,'rb')); base=pickle.load(open(args.base,'rb'))
    orig=[l.rstrip('\n') for l in open(HOLD+'/holdout_orig.txt')]
    gapf=[l.rstrip('\n') for l in open(HOLD+'/holdout_gap.txt')]
    meta=[l.rstrip('\n').split('\t') for l in open(HOLD+'/holdout_meta.tsv')][1:]
    W=np.array([float(x) for x in args.weights.split(',')]); Ws=W.sum()
    n=len(gp)
    # precompute distributions
    recs=[]
    for i in range(n):
        a=np.asarray(gp[i]); M=len(toks[i])
        if len(a)==0: recs.append(None); continue
        j=int(np.argmax(a)); k=j+1; p=float(a[j])
        if k<1 or k>M-1: recs.append(None); continue
        comb={}
        for dump,wt in zip([zs,ft,base],W):
            if wt<=0: continue
            g=dump[i].get(k) if i<len(dump) else None
            if g is None: continue
            d=wd(g,args.topk)
            for w,v in d.items(): comb[w]=comb.get(w,0.0)+wt*v/Ws
        recs.append((k,p,comb))
    base_costs=[lev(gapf[i],orig[i]) for i in range(n)]
    print('n',n,'baseline',np.mean(base_costs))
    for pc in [1.0,0.7,0.5,0.3]:
        for risk in [1.0,0.8,0.6,1.2]:
            tot=0.0; app=0; loc=0; we=0
            for i in range(n):
                r=recs[i]
                if r is None or not r[2]:
                    tot+=lev(gapf[i],orig[i]); continue
                k,p,comb=r; pp=p**pc
                items=sorted(comb.items(), key=lambda kv:-kv[1])[:args.topk]
                ws=[w for w,_ in items]; ps=np.array([x for _,x in items])
                if args.temp!=1.0:
                    lg=np.log(ps+1e-12)/args.temp; ps=np.exp(lg-lg.max())
                ps/=ps.sum()
                best=None
                for c in ws:
                    e=float(sum(ps[jj]*lev(c,ws[jj]) for jj in range(len(ws))))
                    cost=pp*e + risk*(1-pp)*(args.B + len(c) + 1)
                    if best is None or cost<best[0]: best=(cost,c)
                if best[0] < args.B:
                    w=best[1]; tokss=gapf[i].split()
                    pred=' '.join(tokss[:k]+[w]+tokss[k:]); app+=1
                    rmi=int(meta[i][2]); rmw=meta[i][3]
                    if k==rmi: loc+=1
                    if k==rmi and w==rmw: we+=1
                    tot+=lev(pred,orig[i])
                else:
                    tot+=lev(gapf[i],orig[i])
            print(f'pc={pc} risk={risk}: metric {tot/n:.4f} apply {app/n:.3f} loc {loc/max(1,app):.3f} word {we/max(1,app):.3f}')
if __name__=='__main__': main()
