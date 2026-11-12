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

def greedy(names,R,y,grp,mask,maxit=10):
    total=np.zeros(len(y)); cnt=0; sel=[]; best_overall=0
    for it in range(maxit):
        best=(-1,None)
        for n in names:
            sc=(total+R[n])/(cnt+1)
            sm=smooth_sorted(sc,grp,5); m,_=best_mcc(y[mask],sm[mask])
            if m>best[0]: best=(m,n)
        if best[0]<=best_overall+1e-7: break
        best_overall=best[0]; total=total+R[best[1]]; cnt+=1; sel.append(best[1])
    return Counter(sel),best_overall

def main():
    arts=discover(); print('artifacts',list(arts.keys()),flush=True)
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    ref=pd.read_parquet(os.path.join(HERE,'oof_pred_feat3.parquet'),columns=['contact_id']).merge(lab,on='contact_id',how='left')
    ref['game_play'],ref['step'],ref['p1'],ref['p2']=keys_of(ref.contact_id)
    ref=ref.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    grp=ref.groupby(['game_play','p1','p2'],sort=False).ngroup().values
    y=ref.contact.values.astype(np.float64); isg=(ref.p2=='G').values
    R={}
    good={}
    for name,(op,tp) in arts.items():
        try:
            o=pd.read_parquet(op).drop_duplicates('contact_id').set_index('contact_id')
            v=o['prob'].reindex(ref.contact_id.values).values
            v=np.nan_to_num(v,nan=np.nanmedian(v)); R[name]=rankdata(v)/len(v)
            good[name]=(op,tp)
        except Exception as e:
            print('SKIP unreadable',name,repr(e)[:80],flush=True)
    arts=good; names=list(R.keys())
    wg,mg=greedy(names,R,y,grp,isg)
    wp,mp=greedy(names,R,y,grp,~isg)
    print('ground greedy',dict(wg),'ground MCC %.5f'%mg,flush=True)
    print('pair greedy',dict(wp),'pair MCC %.5f'%mp,flush=True)
    def combo(wg,wp):
        bg=sum(R[n]*c for n,c in wg.items())/sum(wg.values())
        bp=sum(R[n]*c for n,c in wp.items())/sum(wp.values())
        return np.where(isg,bg,bp)
    for label,(a,b) in [('pertype',(wg,wp))]:
        sc=combo(a,b)
        for w in (3,5,7):
            sm=smooth_sorted(sc,grp,w); m,th=best_mcc(y,sm)
            print('%s w%d overall MCC %.5f thr %.5f'%(label,w,m,th),flush=True)
    # best overall from pertype search
    best=(-1,None,None,0,0)
    for w in (3,5,7,9):
        sc=combo(wg,wp); sm=smooth_sorted(sc,grp,w); m,th=best_mcc(y,sm)
        if m>best[0]: best=(m,wg,wp,w,th)
    m,wg,wp,w,th=best
    print('BEST pertype %.5f w%d thr %.5f'%(m,w,th),flush=True)
    # test
    ss=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    def tscore(w):
        s=np.zeros(len(ss))
        for n,c in w.items():
            try:
                t=pd.read_parquet(arts[n][1]).drop_duplicates('contact_id').set_index('contact_id')
                tv=t['prob'].reindex(ss.contact_id.values).values
                s+=rankdata(np.nan_to_num(tv,nan=0.0))/len(tv)*c
            except Exception as e:
                print('SKIP test unreadable',n,repr(e)[:80],flush=True)
        return s/sum(w.values())
    td=pd.DataFrame({'contact_id':ss.contact_id}); td['game_play'],td['step'],td['p1'],td['p2']=keys_of(td.contact_id)
    isgt=(td.p2=='G').values
    sc=np.where(isgt,tscore(wg),tscore(wp))
    td=td.assign(prob=sc).sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    g2=td.groupby(['game_play','p1','p2'],sort=False).ngroup().values
    td['sm']=smooth_sorted(td.prob.values,g2,w); td['pred']=(td.sm.values>=th).astype(int)
    out=td.set_index('contact_id')['pred'].reindex(ss.contact_id).reset_index(); out.columns=['contact_id','contact']
    assert out.contact.notna().all() and len(out)==len(ss)
    out.to_csv(os.path.join(HERE,'sub_pertype_fom.csv'),index=False)
    print('test posrate %.4f'%out.contact.mean(),flush=True)
    json.dump(dict(wg=dict(wg),wp=dict(wp),w=w,thr=th,mcc=m),open(os.path.join(HERE,'ens_pertype_fom.json'),'w'))

if __name__=='__main__': main()
