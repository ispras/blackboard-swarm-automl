import pickle,argparse
import numpy as np
from Levenshtein import distance as lev
COMMON=["the","a","of","to","and","in","that","is","for","on","with","as","at","by","from","it","an","be","this","was","or","its","his","her","their","not","but","has","had","are"]
def gen(ws,kind):
    c=set(ws); c.update(COMMON)
    for w in ws:
        for L in range(1,len(w)): c.add(w[:L])
        if kind>=2:
            for L in range(1,len(w)): c.add(w[L:])
            for k in range(len(w)): 
                if len(w)>1: c.add(w[:k]+w[k+1:])
        if kind>=3:
            for a in range(len(w)):
                for b in range(a+2,len(w)+1): c.add(w[a:b])
            for k in range(len(w)):
                for ch in 'etaoinshrdlu':
                    c.add(w[:k]+ch+w[k+1:])
    return c
def choose(ws,ps,T,lam,cands):
    ps=np.array(ps,dtype=float)**(1.0/T); S=ps.sum() or 1; pf=ps/S
    best=None;bs=None
    for s in cands:
        e=sum(pf[i]*lev(s,ws[i]) for i in range(len(ws)))+lam*len(s)
        if best is None or e<best: best=e;bs=s
    return bs
def ins(t,g,w): return ' '.join(t[:g])+' '+str(w)+' '+' '.join(t[g:])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',required=True)
    ap.add_argument('--kind',type=int,default=2); ap.add_argument('--lam',type=float,default=0.05); ap.add_argument('--T',type=float,default=1.25)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    C=pickle.load(open(args.cache,'rb')); post=C['post']; WL=C['WL']; toks=C['tokens']; meta=C['meta']; orig=C['orig']
    N=len(toks); base=[lev(orig[i],' '.join(toks[i])) for i in range(N)]
    if args.limit: N=min(N,args.limit)
    info={}
    ids=range(N) if not args.limit else range(min(args.limit,N))
    for i in ids:
        if i not in post: continue
        q=post[i]; gs=int(q.argmax()); g=gs+1; p=float(q[gs])
        if (i,g) not in WL: continue
        ws,ps=WL[(i,g)]
        if not ws: continue
        c=gen(ws,args.kind); u=choose(ws,ps,args.T,args.lam,c); info[i]=(g,p,u)
    print('kind',args.kind,'T',args.T,'lam',args.lam)
    for tau in [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]:
        tot=0;app=0;lok=0
        for i in range(N):
            if i not in info: tot+=base[i]; continue
            g,p,u=info[i]
            if p<tau: tot+=base[i]; continue
            tot+=lev(ins(toks[i],g,u),orig[i]); app+=1; lok+=(g==int(meta[i][2]))
        print('tau %.2f metric %.4f apply %.3f loc %.3f'%(tau,tot/N,app/N,lok/max(app,1)))
if __name__=='__main__': main()
