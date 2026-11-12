import os, sys, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache'); DATA=os.path.join(HERE,'..','..','data')
WS=os.path.join(HERE,'..')

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def keys_of(cid):
    parts=cid.str.split('_')
    return (parts.str[0]+'_'+parts.str[1]), parts.str[2].astype(int), parts.str[3], parts.str[4]

def discover():
    out={}
    for p in glob.glob(os.path.join(WS,'*','oof_pred*.parquet')):
        name=os.path.basename(os.path.dirname(p))+'_'+os.path.basename(p).replace('oof_pred','').replace('.parquet','').strip('_')
        tp=p.replace('oof_pred','test_pred')
        if os.path.exists(tp):
            out[name]=(p,tp)
    return out

def smooth_sorted(vals, grp_codes, w):
    s=pd.Series(vals)
    r=s.groupby(grp_codes).rolling(w,min_periods=1,center=True).mean()
    return r.values

def main():
    arts=discover()
    print('artifacts:',list(arts.keys()),flush=True)
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    # reference keys from my base
    ref=pd.read_parquet(os.path.join(HERE,'oof_pred.parquet'),columns=['contact_id'])
    ref=ref.merge(lab,on='contact_id',how='left')
    ref['game_play'],ref['step'],ref['p1'],ref['p2']=keys_of(ref.contact_id)
    ref=ref.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    grp=ref.groupby(['game_play','p1','p2'],sort=False).ngroup().values
    y=ref.contact.values.astype(np.float64)
    R={}
    for name,(op,tp) in arts.items():
        o=pd.read_parquet(op).drop_duplicates('contact_id').set_index('contact_id')
        v=o['prob'].reindex(ref.contact_id.values).values
        if np.isnan(v).any():
            print('WARN missing',name,np.isnan(v).sum()); v=np.nan_to_num(v,nan=np.nanmedian(v))
        R[name]=rankdata(v)/len(v)
        print('  loaded',name,flush=True)
    names=list(R.keys())
    # individual
    def ev(score,w=5):
        sm=smooth_sorted(score,grp,w)
        return best_mcc(y,sm)
    for n in names:
        m,t=ev(R[n]); print('single %-40s %.5f'%(n,m),flush=True)
    # greedy
    total=np.zeros(len(ref)); cnt=0; sel=[]
    best_overall=0
    for it in range(12):
        best=(-1,None)
        for n in names:
            cand=(total+R[n])/(cnt+1)
            m,t=ev(cand)
            if m>best[0]: best=(m,n)
        if best[0]<=best_overall+1e-6: break
        best_overall=best[0]; n=best[1]; total=total+R[n]; cnt+=1; sel.append(n)
        print('iter %d add %-40s -> %.5f'%(it,n,best[0]),flush=True)
    from collections import Counter
    wts=Counter(sel); wsum=sum(wts.values())
    print('selected weights:',dict(wts),flush=True)
    blend=sum(R[n]*c for n,c in wts.items())/wsum
    # tune w and threshold
    bestw=(-1,0,0)
    for w in (3,5,7,9,11):
        m,t=ev(blend,w)
        print('final smooth w%d %.5f thr %.5f'%(w,m,t),flush=True)
        if m>bestw[0]: bestw=(m,w,t)
    m,w,th=bestw
    print('BEST ensemble OOF MCC %.5f w%d thr %.5f weights %s'%(m,w,th,dict(wts)),flush=True)
    # test
    ss=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    tblend=np.zeros(len(ss))
    for n,c in wts.items():
        t=pd.read_parquet(arts[n][1]).drop_duplicates('contact_id').set_index('contact_id')
        tv=t['prob'].reindex(ss.contact_id.values).values
        tblend+= (rankdata(np.nan_to_num(tv,nan=0.0))/len(tv))*c
    tblend/=wsum
    td=pd.DataFrame({'contact_id':ss.contact_id,'prob':tblend})
    td['game_play'],td['step'],td['p1'],td['p2']=keys_of(td.contact_id)
    td=td.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    g2=td.groupby(['game_play','p1','p2'],sort=False).ngroup().values
    td['sm']=smooth_sorted(td.prob.values,g2,w)
    td['pred']=(td.sm.values>=th).astype(int)
    out=td.set_index('contact_id')['pred'].reindex(ss.contact_id).reset_index(); out.columns=['contact_id','contact']
    assert out.contact.notna().all() and len(out)==len(ss)
    out.to_csv(os.path.join(HERE,'sub_ens_greedy.csv'),index=False)
    print('test pos rate %.4f'%out.contact.mean(),flush=True)
    json.dump(dict(weights=dict(wts),mcc=m,w=w,thr=th),open(os.path.join(CACHE,'ens_greedy.json'),'w'))
    print('wrote',os.path.join(HERE,'sub_ens_greedy.csv'),flush=True)

if __name__=='__main__':
    main()
