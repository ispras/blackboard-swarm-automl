import os, sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from collections import Counter

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache'); DATA=os.path.join(HERE,'..','..','data')
WS=os.path.join(HERE,'..')

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys); P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0); i=int(np.argmax(mcc)); return float(mcc[i]),float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def keys_of(cid):
    parts=cid.str.split('_'); return (parts.str[0]+'_'+parts.str[1]), parts.str[2].astype(int), parts.str[3], parts.str[4]

def discover():
    out={}
    for p in glob.glob(os.path.join(WS,'*','oof_pred*.parquet')):
        name=os.path.basename(os.path.dirname(p))+'_'+os.path.basename(p).replace('oof_pred','').replace('.parquet','').strip('_')
        tp=p.replace('oof_pred','test_pred')
        if os.path.exists(tp): out[name]=(p,tp)
    return out

def smooth_sorted(vals,grp,w):
    return pd.Series(vals).groupby(grp).rolling(w,min_periods=1,center=True).mean().values

def score_of(w,R):
    return sum(R[n]*c for n,c in w.items())/max(1,sum(w.values()))

def plain(names,R,y,grp,mask,maxit=10):
    total=np.zeros(len(y)); cnt=0; sel=[]; bo=0
    for it in range(maxit):
        best=(-1,None)
        for n in names:
            sm=smooth_sorted((total+R[n])/(cnt+1),grp,5); m,_=best_mcc(y[mask],sm[mask])
            if m>best[0]: best=(m,n)
        if best[0]<=bo+1e-7: break
        bo=best[0]; total=total+R[best[1]]; cnt+=1; sel.append(best[1])
    return Counter(sel)

def bagged(names,R,y,grp,mask,nbags=6,maxit=6,frac=0.65):
    rng=np.random.RandomState(0); cnts=Counter()
    for b in range(nbags):
        sub=list(rng.choice(names,size=max(3,int(len(names)*frac)),replace=False))
        tot=np.zeros(len(y)); cnt=0; bo=0
        for it in range(maxit):
            best=(-1,None)
            for n in sub:
                sm=smooth_sorted((tot+R[n])/(cnt+1),grp,5); m,_=best_mcc(y[mask],sm[mask])
                if m>best[0]: best=(m,n)
            if best[0]<=bo+1e-7: break
            bo=best[0]; tot=tot+R[best[1]]; cnt+=1; cnts[best[1]]+=1
    return cnts

def main():
    arts=discover(); print('artifacts',list(arts.keys()),flush=True)
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    ref=pd.read_parquet(os.path.join(HERE,'oof_pred.parquet'),columns=['contact_id']).merge(lab,on='contact_id',how='left')
    ref['game_play'],ref['step'],ref['p1'],ref['p2']=keys_of(ref.contact_id)
    ref=ref.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    grp=ref.groupby(['game_play','p1','p2'],sort=False).ngroup().values
    y=ref.contact.values.astype(np.float64); isg=(ref.p2=='G').values
    R={}
    for name,(op,tp) in arts.items():
        o=pd.read_parquet(op).drop_duplicates('contact_id').set_index('contact_id')
        v=np.nan_to_num(o['prob'].reindex(ref.contact_id.values).values,nan=0.0); R[name]=rankdata(v)/len(v)
    names=list(R.keys())
    res={}
    for tag,fn in [('plain',plain),('bag',bagged)]:
        wg=fn(names,R,y,grp,isg); wp=fn(names,R,y,grp,~isg)
        sc=np.where(isg,score_of(wg,R),score_of(wp,R))
        best=(-1,0,0)
        for w in (3,5,7):
            sm=smooth_sorted(sc,grp,w); m,th=best_mcc(y,sm)
            if m>best[0]: best=(m,w,th)
        mg,_=best_mcc(y[isg], smooth_sorted(score_of(wg,R),grp,5)[isg])
        mp,_=best_mcc(y[~isg], smooth_sorted(score_of(wp,R),grp,5)[~isg])
        print('%s: overall %.5f w%d thr %.5f | ground %.5f pair %.5f'%(tag,best[0],best[1],best[2],mg,mp),flush=True)
        print('  wg',dict(wg),flush=True); print('  wp',dict(wp),flush=True)
        res[tag]=(best,wg,wp)
    tag=max(res,key=lambda k:res[k][0][0]); best,wg,wp=res[tag]
    print('CHOSEN',tag,'overall %.5f'%best[0],flush=True)
    m,w,th=best
    ss=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    def tscore(w):
        s=np.zeros(len(ss))
        for n,c in w.items():
            t=pd.read_parquet(arts[n][1]).drop_duplicates('contact_id').set_index('contact_id')
            tv=np.nan_to_num(t['prob'].reindex(ss.contact_id.values).values,nan=0.0); s+=rankdata(tv)/len(tv)*c
        return s/max(1,sum(w.values()))
    td=pd.DataFrame({'contact_id':ss.contact_id}); td['game_play'],td['step'],td['p1'],td['p2']=keys_of(td.contact_id)
    isgt=(td.p2=='G').values; sc=np.where(isgt,tscore(wg),tscore(wp))
    td=td.assign(prob=sc).sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    g2=td.groupby(['game_play','p1','p2'],sort=False).ngroup().values
    td['sm']=smooth_sorted(td.prob.values,g2,w); td['pred']=(td.sm.values>=th).astype(int)
    out=td.set_index('contact_id')['pred'].reindex(ss.contact_id).reset_index(); out.columns=['contact_id','contact']
    assert out.contact.notna().all() and len(out)==len(ss)
    out.to_csv(os.path.join(HERE,'sub_pertype_bag.csv'),index=False)
    print('test posrate %.4f tag %s'%(out.contact.mean(),tag),flush=True)
    json.dump(dict(tag=tag,overall=m,w=w,thr=th,wg=dict(wg),wp=dict(wp)),open(os.path.join(CACHE,'ens_pertype_bag.json'),'w'))

if __name__=='__main__': main()
