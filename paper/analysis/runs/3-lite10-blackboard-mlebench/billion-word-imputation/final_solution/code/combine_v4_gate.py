import pickle, numpy as np, argparse, csv
from functools import lru_cache
from Levenshtein import distance as _lev
COMMON=["the","a","an","to","of","and","in","that","is","for","on","it","was","with","as","at","by","be","this","from"]
@lru_cache(maxsize=None)
def lev(a,b): return _lev(a,b)
def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}
def choose_cw(comb, level=1.0, lam=0.08, topk=20):
    items=sorted(comb.items(), key=lambda kv:-kv[1])[:topk]
    ws=[w for w,_ in items]; ps=np.array([x for _,x in items],dtype=np.float64)
    if level!=1.0:
        lg=np.log(ps+1e-12)/level; ps=np.exp(lg-lg.max())
    ps/=ps.sum()
    cands=set(ws); cands.update(COMMON)
    for w in ws:
        n=len(w)
        for L in range(1,n): cands.add(w[:L])
        for L in range(1,n): cands.add(w[L:])
        for i in range(n): cands.add(w[:i]+w[i+1:])
    best=None;bs=None;bcw=None
    for c in cands:
        e=float(sum(ps[j]*lev(c,ws[j]) for j in range(len(ws))))
        tot=e+lam*len(c)
        if best is None or tot<best: best=tot; bs=c; bcw=e
    return bs, bcw
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zs',required=True); ap.add_argument('--ft',required=True); ap.add_argument('--base',required=True); ap.add_argument('--alb',required=True)
    ap.add_argument('--gaps',required=True); ap.add_argument('--conf',required=True)
    ap.add_argument('--weights',default='0.38,0.22,0.15,0.25')
    ap.add_argument('--level',type=float,default=1.0)
    ap.add_argument('--lam',type=float,default=0.08)
    ap.add_argument('--tau',type=float,default=0.50)
    ap.add_argument('--tau_lo',type=float,default=0.30)
    ap.add_argument('--cw_thr',type=float,default=0.35)
    ap.add_argument('--topk',type=int,default=20)
    ap.add_argument('--decision_out',default='decisions_v4.tsv')
    ap.add_argument('--submission_out',default='submission_v4.csv')
    ap.add_argument('--test',default='/download_job_code/payload/sandbox/workspaces/spica_psi/test_v2.txt')
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    dumps=[pickle.load(open(p,'rb')) for p in [args.zs,args.ft,args.base,args.alb]]
    conf=np.load(args.conf) if args.conf.endswith('.npy') else pickle.load(open(args.conf,'rb'))
    gaps=pickle.load(open(args.gaps,'rb'))
    W=np.array([float(x) for x in args.weights.split(',')]); Ws=W.sum()
    n=min(len(conf),len(gaps),*[len(d) for d in dumps])
    if args.limit: n=min(n,args.limit)
    ids=[];sents=[]
    with open(args.test,newline='') as f:
        r=csv.reader(f); next(r)
        for row in r:
            if len(row)>=2: ids.append(int(row[0])); sents.append(row[1])
    nd=0;fsub=open(args.submission_out,'w',newline='');wr=csv.writer(fsub,quoting=csv.QUOTE_NONNUMERIC);wr.writerow(['id','sentence'])
    with open(args.decision_out,'w') as fd:
        for i in range(n):
            cf=float(conf[i]); pred=sents[i]; g=gaps[i]
            k=int(g) if isinstance(g,(int,np.integer)) else (int(g[0]) if g else 0)
            do=False
            if k>=1 and len(sents[i].split())>k:
                if cf>=args.tau: do=True
                elif cf>=args.tau_lo:
                    # need cw
                    do=None  # compute below
            if cf>=args.tau: do=True
            if k>=1 and (cf>=args.tau or (cf>=args.tau_lo)):
                comb={}
                for dump,wt in zip(dumps,W):
                    gd=dump[i].get(k) if i<len(dump) else None
                    if gd is None: continue
                    d={}
                    for wq,pp in zip(gd['w'][:args.topk],gd['p'][:args.topk]): d[wq]=d.get(wq,0.0)+pp
                    d=norm(d)
                    for wq,v in d.items(): comb[wq]=comb.get(wq,0.0)+wt*v/Ws
                if comb:
                    s_star,cw=choose_cw(comb,args.level,args.lam,args.topk)
                    if cf>=args.tau or (cf>=args.tau_lo and cw<=args.cw_thr):
                        toks=sents[i].split()
                        if 1<=k<=len(toks)-1:
                            fd.write(f'{ids[i]}\t{k}\t{s_star}\n'); nd+=1
                            pred=' '.join(toks[:k]+[s_star]+toks[k:])
            wr.writerow([ids[i],pred])
    fsub.close()
    print('decisions',nd,'->',args.decision_out)
if __name__=='__main__': main()
