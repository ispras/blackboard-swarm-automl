import pickle, numpy as np, sys, argparse
from Levenshtein import distance as lev
HOLD='/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout'
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--tri', default='tri_holdout_allgaps.pkl')
    args=ap.parse_args()
    d=pickle.load(open(args.dump,'rb'))
    tri=pickle.load(open(args.tri,'rb'))
    meta=[l.rstrip('\n').split('\t') for l in open(HOLD+'/holdout_meta.tsv')][1:]
    n=len(d)
    samples=[]
    for i in range(n):
        rmi=int(meta[i][2]); rmw=meta[i][3]
        if rmi<0: continue
        g=d[i].get(rmi)
        if g is None: continue
        ws=g['w'][:20]; ps=np.array(g['p'][:20],dtype=np.float64); ps=ps/ps.sum()
        tw=tri[i][rmi][0] if (i<len(tri) and rmi in tri[i]) else None
        cand=list(ws)
        if tw is not None and tw not in cand: cand.append(tw)
        L=np.array([[lev(c,w) for w in ws] for c in cand],dtype=np.float64)
        lr=np.array([lev(c,rmw) for c in cand],dtype=np.float64)
        samples.append((ws,ps,cand,L,lr))
    print('samples',len(samples))
    def ev(temp=1.0,lam=0.0,topk=20,use_tri=False):
        tot=0.0
        for ws,ps,cand,L,lr in samples:
            kk=min(topk,len(ps)); p=ps[:kk].copy()
            if temp!=1.0:
                lg=np.log(p+1e-12)/temp; p=np.exp(lg-lg.max()); p/=p.sum()
            else: p/=p.sum()
            cidx=list(range(kk))
            if use_tri and len(cand)>len(ws): cidx.append(len(ws))
            best=None
            for ci in cidx:
                e=float(np.dot(p,L[ci,:kk]))+lam*len(cand[ci])
                if best is None or e<best[0]: best=(e,ci)
            tot+=lr[best[1]]
        return tot/len(samples)
    print('top1', round(ev_temp:=float(np.mean([lr[0] for ws,ps,cand,L,lr in samples])),4))
    rows=[]
    for temp in [1.0,1.25,1.5,1.75,2.0]:
        for lam in [0.0,0.05,0.1,0.2]:
            for tk in [5,10,20]:
                for tr in [False,True]:
                    rows.append((ev(temp,lam,tk,tr),temp,lam,tk,tr))
    rows.sort()
    for r in rows[:10]: print(f'cost={r[0]:.4f} temp={r[1]} lam={r[2]} topk={r[3]} tri={r[4]}')
if __name__=='__main__': main()
