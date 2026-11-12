import pickle, numpy as np, argparse, csv
from Levenshtein import distance as lev

def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}

def word_dist(g, k=20):
    ws=g['w'][:k]; ps=g['p'][:k]
    d={}
    for w,p in zip(ws,ps): d[w]=d.get(w,0.0)+p
    return norm(d)

def choose_word(comb, temp=1.5, lam=0.05, topk=20):
    items=sorted(comb.items(), key=lambda kv:-kv[1])[:topk]
    ws=[w for w,_ in items]; ps=np.array([p for _,p in items])
    if temp!=1.0:
        lg=np.log(ps+1e-12)/temp; ps=np.exp(lg-lg.max())
    ps/=ps.sum()
    best=None
    for c in ws:
        e=float(sum(ps[j]*lev(c,ws[j]) for j in range(len(ws))))+lam*len(c)
        if best is None or e<best[0]: best=(e,c)
    return best[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zs', required=True)
    ap.add_argument('--ft', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--conf', required=True, help='pkl/npy list of detector confidence per sentence; None/negative = no gap')
    ap.add_argument('--tau', type=float, default=0.55)
    ap.add_argument('--weights', default='0.5,0.3,0.2')
    ap.add_argument('--temp', type=float, default=1.5)
    ap.add_argument('--lam', type=float, default=0.05)
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--decision_out', default='decisions.tsv')
    ap.add_argument('--submission_out', default='')
    ap.add_argument('--test', default='/download_job_code/payload/sandbox/workspaces/spica_psi/test_v2.txt')
    ap.add_argument('--limit', type=int, default=0)
    args=ap.parse_args()
    zs=pickle.load(open(args.zs,'rb')); ft=pickle.load(open(args.ft,'rb')); base=pickle.load(open(args.base,'rb'))
    if args.conf.endswith('.npy'): conf=np.load(args.conf, allow_pickle=True)
    else: conf=pickle.load(open(args.conf,'rb'))
    W=np.array([float(x) for x in args.weights.split(',')]); Ws=W.sum()
    n=min(len(zs),len(ft),len(base),len(conf))
    if args.limit: n=min(n,args.limit)
    # read test
    ids=[]; sents=[]
    with open(args.test, newline='') as f:
        rd=csv.reader(f); next(rd)
        for row in rd:
            if len(row)>=2: ids.append(int(row[0])); sents.append(row[1])
    nd=0
    with open(args.decision_out,'w') as fd:
        fsub=None; wr=None
        if args.submission_out:
            fsub=open(args.submission_out,'w',newline=''); wr=csv.writer(fsub, quoting=csv.QUOTE_NONNUMERIC); wr.writerow(['id','sentence'])
        for i in range(n):
            c=conf[i]
            try: cf=float(c)
            except Exception: cf=-1.0
            pred=sents[i]
            if cf>=args.tau:
                comb={}
                anyg=False
                for dump,wt in zip([zs,ft,base],W):
                    if wt<=0: continue
                    # each dump[i] should have exactly one gap key (chosen)
                    if not dump[i]: continue
                    gk=next(iter(dump[i].keys()))
                    d=word_dist(dump[i][gk],args.topk)
                    for w,v in d.items(): comb[w]=comb.get(w,0.0)+wt*v/Ws
                    anyg=True
                if comb:
                    w=choose_word(comb,args.temp,args.lam,args.topk)
                    gk=next(iter(zs[i].keys())) if zs[i] else (next(iter(base[i].keys())) if base[i] else None)
                    if gk is not None:
                        toks=sents[i].split()
                        if 1<=gk<=len(toks)-1:
                            fd.write(f'{ids[i]}\t{gk}\t{w}\n'); nd+=1
                            pred=' '.join(toks[:gk]+[w]+toks[gk:])
            if wr is not None:
                wr.writerow([ids[i], pred])
        if fsub: fsub.close()
    print('decisions written',nd,'to',args.decision_out)
    if args.submission_out: print('submission written',args.submission_out)

if __name__=='__main__': main()
