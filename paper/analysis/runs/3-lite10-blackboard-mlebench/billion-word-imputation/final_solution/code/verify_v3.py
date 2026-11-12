import pickle,argparse
import numpy as np
from Levenshtein import distance as lev
COMMON=["the","a","of","to","and","in","that","is","for","on","with","as","at","by","from","it","an","be","this","was","or","its","his","her","their","not","but","has","had","are"]
def fuse(ms,weights,eps=1e-6):
    d={}
    for k,(m) in enumerate(ms):
        if not m: continue
        for w,p in zip(m['w'],m['p']):
            if w not in d: d[w]=[np.log(eps)]*len(ms)
            d[w][k]=weights[k]*np.log(max(p,eps))
    items=sorted([(w,sum(v)) for w,v in d.items()],key=lambda x:-x[1])[:25]
    mx=items[0][1]; ex=np.exp(np.array([s for w,s in items])-mx); return [w for w,s in items],(ex/ex.sum()).tolist()
def gen(ws):
    c=set(ws); c.update(COMMON)
    for w in ws:
        for L in range(1,len(w)): c.add(w[:L]); c.add(w[L:])
        if len(w)>1:
            for k in range(len(w)): c.add(w[:k]+w[k+1:])
    return c
def choose(ws,ps,T,lam):
    ps=np.array(ps,dtype=float)**(1.0/T); S=ps.sum() or 1; pf=ps/S
    c=gen(ws); best=None;bs=None
    for s in c:
        e=sum(pf[i]*lev(s,ws[i]) for i in range(len(ws)))+lam*len(s)
        if best is None or e<best: best=e;bs=s
    return bs
def ins(t,g,w): return ' '.join(t[:g])+' '+str(w)+' '+' '.join(t[g:])
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--chosen',default='hold_chosen_gaps_mix3.pkl'); ap.add_argument('--conf',default='hold_conf_mix3.npy')
    ap.add_argument('--zs',default='/download_job_code/payload/sandbox/workspaces/spica_psi/hold_mix3_zs.pkl')
    ap.add_argument('--ft',default='/download_job_code/payload/sandbox/workspaces/spica_psi/hold_mix3_ft.pkl')
    ap.add_argument('--base',default='/download_job_code/payload/sandbox/workspaces/spica_psi/hold_mix3_base.pkl')
    ap.add_argument('--gapf',default='/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout/holdout_gap.txt')
    ap.add_argument('--orig',default='/download_job_code/payload/sandbox/workspaces/antares_upsilon/holdout/holdout_orig.txt')
    ap.add_argument('--T',type=float,default=1.25); ap.add_argument('--lam',type=float,default=0.05)
    ap.add_argument('--w',default='0.5,0.3,0.2')
    args=ap.parse_args()
    gaps=pickle.load(open(args.chosen,'rb')); conf=np.load(args.conf)
    toks=[l.rstrip('\n').split() for l in open(args.gapf)]; orig=[l.rstrip('\n') for l in open(args.orig)]
    Z=pickle.load(open(args.zs,'rb')); F=pickle.load(open(args.ft,'rb')); B=pickle.load(open(args.base,'rb'))
    W=[float(x) for x in args.w.split(',')]
    N=len(toks); base=[lev(orig[i],' '.join(toks[i])) for i in range(N)]
    # precompute strings (tau-independent)
    chosen=[None]*N
    for i in range(N):
        g=gaps[i]
        if not g or g<1 or g>len(toks[i])-1: continue
        d={}
        ms=[ (Z[i].get(g) if i<len(Z) else None), (F[i].get(g) if i<len(F) else None), (B[i].get(g) if i<len(B) else None) ]
        ws,ps=fuse(ms,W)
        if not ws: continue
        chosen[i]=choose(ws,ps,args.T,args.lam)
    print('baseline',round(np.mean(base),4),'T',args.T,'lam',args.lam,'w',W)
    best=None
    for tau in [0.40,0.45,0.50,0.55,0.60,0.65,0.70]:
        tot=0;app=0;lok=0;wok=0
        for i in range(N):
            if chosen[i] is None or conf[i]<tau: tot+=base[i]; continue
            g=gaps[i]; tot+=lev(ins(toks[i],g,chosen[i]),orig[i]); app+=1
        m=tot/N
        print('tau %.2f metric %.4f apply %.3f'%(tau,m,app/N))
        if best is None or m<best[0]: best=(m,tau)
    print('BEST',best)
if __name__=='__main__': main()
