import pickle,argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',default='gate_cache_mix3.pkl')
    args=ap.parse_args()
    R=pickle.load(open(args.cache,'rb')); N=len(R)
    base=np.array([ (r['base'] if r else 0) for r in R],dtype=float)
    idx=[i for i,r in enumerate(R) if r is not None]
    conf=np.array([R[i]['conf'] for i in idx]); marg=np.array([R[i]['margin'] for i in idx])
    ngap=np.array([R[i]['ngap'] for i in idx],dtype=float); cw=np.array([R[i]['cw'] for i in idx])
    sl=np.array([len(R[i]['s']) for i in idx],dtype=float)
    cost=np.array([R[i]['cost'] for i in idx],dtype=float); bas=np.array([R[i]['base'] for i in idx],dtype=float)
    benefit=bas-cost
    print('n',len(idx),'baseline all',base.mean(),'insert-all',(base.sum()-benefit.sum())/N)
    def metric(mask):
        tot=base.sum()
        # for inserted: replace base with cost
        tot=base.sum() - np.where(mask,benefit,0).sum()
        return tot/N
    print('--- conf gate ---')
    for t in [0.4,0.45,0.48,0.5,0.52,0.55,0.6]:
        m=conf>=t; print('t',t,'metric',round(metric(m),4),'apply',round(m.mean(),3))
    print('--- conf+a*margin ---')
    for a in [0.1,0.2,0.3,0.5]:
        s=conf+a*marg
        best=None
        for t in np.linspace(0.3,0.8,26):
            m=s>=t; v=metric(m)
            if best is None or v<best[0]: best=(v,t,m.mean())
        print('a',a,'best',round(best[0],4),'t',round(best[1],3),'apply',round(best[2],3))
    print('--- word-aware OR ---')
    best=None
    for c0 in [0.3,0.5,0.8,1.0,1.5]:
        for t1 in [0.25,0.3,0.35,0.4]:
            m=(conf>=0.5)|((cw<=c0)&(conf>=t1)); v=metric(m)
            if best is None or v<best[0]: best=(v,c0,t1,m.mean())
    print('best OR',best)
    # GBM OOF predicting benefit
    X=np.column_stack([conf,marg,np.log(ngap),cw,sl,conf*marg,conf/np.sqrt(ngap)])
    pred=np.zeros(len(idx))
    kf=KFold(5,shuffle=True,random_state=0)
    for tr,te in kf.split(X):
        g=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05,l2_regularization=1.0)
        g.fit(X[tr],benefit[tr]); pred[te]=g.predict(X[te])
    print('--- GBM OOF benefit threshold ---')
    best=None
    for t in np.linspace(-0.5,1.5,41):
        m=pred>t; v=metric(m)
        if best is None or v<best[0]: best=(v,t,m.mean())
    print('GBM best',best)
    # ridge
    pred2=np.zeros(len(idx))
    for tr,te in kf.split(X):
        g=Ridge(alpha=10.0); g.fit(X[tr],benefit[tr]); pred2[te]=g.predict(X[te])
    best=None
    for t in np.linspace(-0.5,1.5,41):
        m=pred2>t; v=metric(m)
        if best is None or v<best[0]: best=(v,t,m.mean())
    print('Ridge best',best)
if __name__=='__main__': main()
