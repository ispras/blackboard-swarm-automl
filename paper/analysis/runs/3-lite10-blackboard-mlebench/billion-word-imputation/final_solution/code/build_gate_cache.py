import pickle,argparse
import numpy as np
from Levenshtein import distance as lev
COMMON=["the","a","an","to","of","and","in","that","is","for","on","it","was","with","as","at","by","be","this","from"]
def norm(d):
    s=sum(d.values()); return {w:v/s for w,v in d.items()} if s>0 else {}
def make_cands(ws,kind=2):
    c=set(ws); c.update(COMMON)
    for w in ws:
        n=len(w)
        for L in range(1,n): c.add(w[:L])
        if kind>=2:
            for L in range(1,n): c.add(w[L:])
            for i in range(n): c.add(w[:i]+w[i+1:])
    return list(c)
def choose(comb,level=1.0,lam=0.05,topk=20,kind=2):
    items=sorted(comb.items(),key=lambda kv:-kv[1])[:topk]
    ws=[w for w,_ in items]; ps=np.array([x for _,x in items],dtype=float)
    lg=np.log(ps+1e-12)/level; ps=np.exp(lg-lg.max()); ps/=ps.sum()
    cands=make_cands(ws,kind); best=None;bw=None
    for c in cands:
        e=float(sum(ps[j]*lev(c,ws[j]) for j in range(len(ws))))+lam*len(c)
        if best is None or e<best: best=e;bw=c
    return bw,ws,ps
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--det',default='det_hold_mix3_arith.pkl')
    ap.add_argument('--zs',default='/download_job_code/payload/sandbox/workspaces/spica_psi/hold_mix3_zs.pkl')
    ap.add_argument('--ft',default='/download_job_code/payload/sandbox/workspaces/spica_psi/hold_mix3_ft.pkl')
    ap.add_argument('--base',default='/download_job_code/payload/sandbox/workspaces/spica_psi/hold_mix3_base.pkl')
    ap.add_argument('--out',default='gate_cache_mix3.pkl')
    args=ap.parse_args()
    D=pickle.load(open(args.det,'rb')); gp=D['gap_probs']; toks=D['tokens']
    Z=pickle.load(open(args.zs,'rb')); F=pickle.load(open(args.ft,'rb')); B=pickle.load(open(args.base,'rb'))
    orig=[l.rstrip('\n') for l in open('/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout/holdout_orig.txt')]
    meta=[l.rstrip('\n').split('\t') for l in open('/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout/holdout_meta.tsv')][1:]
    W=[0.5,0.3,0.2]; Ws=sum(W)
    recs=[]
    for i in range(len(toks)):
        a=np.asarray(gp[i]); M=len(toks[i])
        if len(a)==0: recs.append(None); continue
        j=int(np.argmax(a)); k=j+1; p=float(a[j])
        if k<1 or k>M-1: recs.append(None); continue
        srt=np.sort(a); margin=float(srt[-1]-srt[-2]) if len(a)>1 else float(srt[-1])
        comb={}
        for dump,wt in zip([Z,F,B],W):
            g=dump[i].get(k) if i<len(dump) else None
            if g is None: continue
            d=norm({w:pp for w,pp in zip(g['w'][:20],g['p'][:20])})
            for w,v in d.items(): comb[w]=comb.get(w,0.0)+wt*v/Ws
        if not comb: recs.append(None); continue
        bw,ws,ps=choose(comb)
        # expected word cost of chosen string under fused dist
        cw=float(sum(ps[jj]*lev(bw,ws[jj]) for jj in range(len(ws))))
        pred=' '.join(toks[i][:k]+[bw]+toks[i][k:]); cost=lev(pred,orig[i])
        base=lev(orig[i],' '.join(toks[i]))
        loc=(k==int(meta[i][2]))
        recs.append({'i':i,'g':k,'conf':p,'margin':margin,'ngap':M-1,'cw':cw,'s':bw,'cost':cost,'base':base,'loc':loc,'n':M})
        if i%5000==0: print('prep',i,flush=True)
    pickle.dump(recs,open(args.out,'wb'),protocol=4); print('saved',args.out)
if __name__=='__main__': main()
