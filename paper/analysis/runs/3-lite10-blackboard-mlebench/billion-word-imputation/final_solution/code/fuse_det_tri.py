import pickle,sys,argparse
import numpy as np
from Levenshtein import distance as lev
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

def choose_word(words,probs,lam=0.2):
    k=len(words)
    if k==0: return None
    S=sum(probs)
    if S<=0: return words[0]
    pf=[p/S for p in probs]
    best=None;bestu=None
    for u in words:
        e=sum(pf[i]*lev(u,words[i]) for i in range(k))+lam*len(u)
        if best is None or e<best: best=e;bestu=u
    return bestu

def ins(t,g,w): return ' '.join(t[:g])+' '+w+' '+' '.join(t[g:])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dump',required=True)
    ap.add_argument('--tri',default='/download_job_code/payload/sandbox/workspaces/spica_psi/tri_holdout_allgaps.pkl')
    ap.add_argument('--lam',type=float,default=0.2)
    ap.add_argument('--mode',default='both') # det|tri|both
    args=ap.parse_args()
    D=pickle.load(open(args.dump,'rb')); toks=D['tokens']; res=D['res']; meta=D['meta']; orig=D['orig']
    TRI=pickle.load(open(args.tri,'rb'))
    N=len(toks)
    # build per-sentence gap feature arrays
    feats=[];labels=[];gaps_meta=[]
    for i in range(N):
        t=toks[i]; n=len(t); r=res[i]; gp=r['gap_probs']
        if len(gp)==0 or len(gp)!=n-1: feats.append(None); labels.append(None); gaps_meta.append(None); continue
        ng=n-1
        tri=TRI[i] if i<len(TRI) else {}
        # build tri gain per gap
        tg=np.full(ng,-30.0); tp=np.zeros(ng)
        for k,(w,gain,ls) in tri.items():
            if 1<=k<=ng:
                tg[k-1]=gain; tp[k-1]=1.0
        detwp=np.array([r['probs'][g][0] if r['probs'][g] else 0.0 for g in range(len(gp))])
        # ranks
        detrank=np.argsort(np.argsort(-gp))
        trirank=np.argsort(np.argsort(-tg))
        frac=np.arange(1,ng+1)/n
        gpr=np.clip(gp,1e-6,1)
        X=np.column_stack([gpr,np.log(gpr),-detrank,np.clip(tg,-30,5),tp,-trirank,frac,detwp])
        if args.mode=='det': X=X[:,[0,1,2,6,7]]
        elif args.mode=='tri': X=X[:,[3,4,5,6]]
        rm=int(meta[i][2])-1  # gap index = rm_idx-1
        y=(np.arange(1,ng+1)==int(meta[i][2])).astype(int)
        if 0<=rm<ng:
            feats.append(X); labels.append(y); gaps_meta.append((int(meta[i][2]),meta[i][3]))
        else:
            feats.append(None);labels.append(None);gaps_meta.append(None)
    groups=np.arange(N)
    # cross-val OOF predictions
    idxs=[i for i in range(N) if feats[i] is not None]
    Xall=np.vstack([feats[i] for i in idxs]); yall=np.hstack([labels[i] for i in idxs])
    gall=np.repeat(np.array(idxs),[len(labels[i]) for i in idxs])
    local_of_row=np.concatenate([np.arange(len(labels[i])) for i in idxs])
    gkf=GroupKFold(n_splits=3)
    out_p=[None]*N
    for tr,te in gkf.split(Xall,yall,gall):
        sc=StandardScaler().fit(Xall[tr])
        clf=LogisticRegression(max_iter=2000,C=1.0).fit(sc.transform(Xall[tr]),yall[tr])
        pr=clf.predict_proba(sc.transform(Xall[te]))[:,1]
        for j,gi in enumerate(te):
            s_=gall[gi]
            if out_p[s_] is None: out_p[s_]={}
            out_p[s_][int(local_of_row[gi])]=pr[j]
    post=[None]*N
    for s_ in range(N):
        if feats[s_] is None: continue
        arr=np.zeros(len(feats[s_]))
        for gi,v in out_p[s_].items(): arr[gi]=v
        post[s_]=arr/arr.sum()
    # evaluate
    base=[lev(orig[i],' '.join(toks[i])) for i in range(N)]
    print('baseline',np.mean(base))
    for tau in [0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
        tot=0;app=0;lok=0
        for i in range(N):
            if post[i] is None: tot+=base[i]; continue
            gs=int(post[i].argmax()); pb=float(post[i][gs])
            if pb>=tau:
                r=res[i]; g=gs; words=r['words'][g]; probs=r['probs'][g]
                u=choose_word(words,probs,args.lam)
                tot+=lev(ins(toks[i],g+1,u),orig[i]); app+=1; lok+= (g+1==int(meta[i][2]))
            else: tot+=base[i]
        print(f'tau {tau:.2f} apply {app/N:.3f} metric {tot/N:.4f} locprec {lok/max(app,1):.3f}')
if __name__=='__main__': main()
