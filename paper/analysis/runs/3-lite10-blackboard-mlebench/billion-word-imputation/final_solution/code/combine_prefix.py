import pickle, numpy as np, argparse, csv
from Levenshtein import distance as lev
COMMON=["the","a","an","to","of","and","in","that","is","for","on","it","was","with","as","at","by","be","this","from"]
from functools import lru_cache
@lru_cache(maxsize=None)
def clev(a,b): return lev(a,b)
def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}
def choose(comb, level=1.0, lam=0.05, topk=20, kind=1):
    items=sorted(comb.items(), key=lambda kv:-kv[1])[:topk]
    ws=[w for w,_ in items]; ps=np.array([x for _,x in items],dtype=np.float64)
    lg=np.log(ps+1e-12)/level; ps=np.exp(lg-lg.max()); ps/=ps.sum()
    cands=set(ws); cands.update(COMMON)
    for w in ws:
        n=len(w)
        for L in range(1,n): cands.add(w[:L])
        if kind>=2:
            for L in range(1,n): cands.add(w[L:])
            for i in range(n): cands.add(w[:i]+w[i+1:])
    best=None;bw=None
    for c in cands:
        e=float(sum(ps[j]*clev(c,ws[j]) for j in range(len(ws))))+lam*len(c)
        if best is None or e<best: best=e;bw=c
    return bw
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zs',required=True); ap.add_argument('--ft',required=True); ap.add_argument('--base',required=True)
    ap.add_argument('--conf',required=True)
    ap.add_argument('--tau',type=float,default=0.50)
    ap.add_argument('--level',type=float,default=1.0)
    ap.add_argument('--lam',type=float,default=0.05)
    ap.add_argument('--kind',type=int,default=2)
    ap.add_argument('--weights',default='0.5,0.3,0.2')
    ap.add_argument('--topk',type=int,default=20)
    ap.add_argument('--decision_out',default='decisions_v2.tsv')
    ap.add_argument('--submission_out',default='')
    ap.add_argument('--test',default='/download_job_code/payload/sandbox/workspaces/spica_psi/test_v2.txt')
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    zs=pickle.load(open(args.zs,'rb')); ft=pickle.load(open(args.ft,'rb')); base=pickle.load(open(args.base,'rb'))
    conf=np.load(args.conf) if args.conf.endswith('.npy') else pickle.load(open(args.conf,'rb'))
    W=np.array([float(x) for x in args.weights.split(',')]); Ws=W.sum()
    n=min(len(zs),len(ft),len(base),len(conf))
    if args.limit: n=min(n,args.limit)
    ids=[];sents=[]
    with open(args.test,newline='') as f:
        r=csv.reader(f); next(r)
        for row in r:
            if len(row)>=2: ids.append(int(row[0])); sents.append(row[1])
    nd=0
    fsub=None;wr=None
    if args.submission_out:
        fsub=open(args.submission_out,'w',newline=''); wr=csv.writer(fsub,quoting=csv.QUOTE_NONNUMERIC); wr.writerow(['id','sentence'])
    with open(args.decision_out,'w') as fd:
        for i in range(n):
            cf=float(conf[i]); pred=sents[i]
            if cf>=args.tau:
                comb={}
                for dump,wt in zip([zs,ft,base],W):
                    if wt<=0 or not dump[i]: continue
                    gk=next(iter(dump[i].keys())); g=dump[i][gk]
                    d={}
                    for w,p in zip(g['w'][:args.topk],g['p'][:args.topk]): d[w]=d.get(w,0.0)+p
                    d=norm(d)
                    for w,v in d.items(): comb[w]=comb.get(w,0.0)+wt*v/Ws
                if comb:
                    bw=choose(comb,args.level,args.lam,args.topk,args.kind)
                    gk=next(iter(zs[i].keys())) if zs[i] else (next(iter(base[i].keys())) if base[i] else None)
                    if gk is not None:
                        toks=sents[i].split()
                        if 1<=gk<=len(toks)-1:
                            fd.write(f'{ids[i]}\t{gk}\t{bw}\n'); nd+=1
                            pred=' '.join(toks[:gk]+[bw]+toks[gk:])
            if wr is not None: wr.writerow([ids[i],pred])
    if fsub: fsub.close()
    print('decisions',nd,'->',args.decision_out)
if __name__=='__main__': main()
