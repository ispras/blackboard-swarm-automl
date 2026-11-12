import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from collections import Counter

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache'); DATA=os.path.join(HERE,'..','..','data')
WS=os.path.join(HERE,'..')

def keys_of(cid):
    p=cid.str.split('_'); return (p.str[0]+'_'+p.str[1]), p.str[2].astype(int), p.str[3], p.str[4]

def discover():
    out={}
    for p in glob.glob(os.path.join(WS,'*','oof_pred*.parquet')):
        name=os.path.basename(os.path.dirname(p))+'_'+os.path.basename(p).replace('oof_pred','').replace('.parquet','').strip('_')
        tp=p.replace('oof_pred','test_pred')
        if os.path.exists(tp): out[name]=(p,tp)
    return out

def smooth_by(df, w, keys):
    df=df.sort_values(list(keys)+['step']).reset_index(drop=True)
    r=df.groupby(list(keys),sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    df['sm']=r.reset_index(level=list(range(len(keys))),drop=True).reindex(df.index).values
    return df

def mcc_grid(yg_sorted_pos, yp_sorted_pos, Ng, Np, Pg, Pp, kgs, kps):
    # cumulative positives at k predicted positives
    kgs=kgs.astype(np.float64); kps=kps.astype(np.float64)
    tpg=yg_sorted_pos[kgs.astype(int)].astype(np.float64); fpg=kgs-tpg
    tpp=yp_sorted_pos[kps.astype(int)].astype(np.float64); fpp=kps-tpp
    TP=tpg[:,None]+tpp[None,:]; FP=fpg[:,None]+fpp[None,:]
    P=float(Pg+Pp); N=float((Ng+Np)-P); FN=P-TP; TN=N-FP
    den=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    mcc=np.divide(TP*TN-FP*FN,den,out=np.zeros_like(TP,dtype=float),where=den>0)
    i=np.unravel_index(np.argmax(mcc),mcc.shape)
    return float(mcc[i]), int(kgs[i[0]]), int(kps[i[1]])

def main():
    # hardcoded per-type weights from best greedy run
    wg={'fomalhaut_omicron_feat5_d2':1,'procyon_theta_gruc':1,'procyon_theta_gru':1,'fomalhaut_omicron_feat5':1}
    wp={'fomalhaut_omicron_feat5':3,'procyon_theta_gallr':1,'procyon_theta_l2':1,'fomalhaut_omicron_feat5_d2':1,'procyon_theta_gruc':1}
    arts=discover(); names=set(wg)|set(wp)
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    ref=pd.read_parquet(os.path.join(HERE,'oof_pred.parquet'),columns=['contact_id']).merge(lab,on='contact_id',how='left')
    ref['game_play'],ref['step'],ref['p1'],ref['p2']=keys_of(ref.contact_id)
    ref=ref.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    R={}
    for n in names:
        o=pd.read_parquet(arts[n][0]).drop_duplicates('contact_id').set_index('contact_id')
        v=np.nan_to_num(o['prob'].reindex(ref.contact_id.values).values,nan=0.0)
        R[n]=rankdata(v)/len(v)
    def blend(w):
        s=np.zeros(len(ref))
        for n,c in w.items(): s+=R[n]*c
        return s/sum(w.values())
    isg=(ref.p2=='G').values
    sg=blend(wg); sp=blend(wp); y=ref.contact.values.astype(np.int64)
    Ng=int(isg.sum()); Np=int((~isg).sum()); Pg=int(y[isg].sum()); Pp=int(y[~isg].sum())
    # ground group keys (game_play,p1), pair keys (game_play,p1,p2)
    base=ref[['game_play','step','p1','p2','contact']].copy()
    def oof_sorted(scores, mask, w, keys):
        d=base[mask].copy(); d['prob']=scores[mask]
        d=smooth_by(d,w,keys)
        order=np.argsort(-d.sm.values); ys=(d.contact.values[order]>0).astype(np.int64)
        return np.cumsum(ys), len(d)
    # precompute smoothing for each w combo
    kg_grid=np.unique(np.linspace(0.02*Ng,0.075*Ng,141).astype(int))
    kp_grid=np.unique(np.linspace(0.005*Np,0.02*Np,181).astype(int))
    best=(-1,None)
    for wgr in (5,7,9):
        cg,Ngg=oof_sorted(sg,isg,wgr,('game_play','p1'))
        for wpr in (3,5,7):
            cp,Npp=oof_sorted(sp,~isg,wpr,('game_play','p1','p2'))
            m,kg,kp=mcc_grid(cg,cp,Ngg,Npp,Pg,Pp,kg_grid,kp_grid)
            qg=kg/Ngg; qp=kp/Npp
            print('wg%d wp%d -> OOF MCC %.5f qg %.4f qp %.4f'%(wgr,wpr,m,qg,qp),flush=True)
            if m>best[0]: best=(m,(wgr,wpr,qg,qp))
    m,(wgr,wpr,qg,qp)=best
    print('BEST OOF %.5f wg%d wp%d qg %.4f qp %.4f'%(m,wgr,wpr,qg,qp),flush=True)
    json.dump(dict(mcc=m,wgr=wgr,wpr=wpr,qg=qg,qp=qp,wg=wg,wp=wp),open(os.path.join(CACHE,'final_opt.json'),'w'))
    # ---- test ----
    ss=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    td=pd.DataFrame({'contact_id':ss.contact_id}); td['game_play'],td['step'],td['p1'],td['p2']=keys_of(td.contact_id)
    Rt={}
    for n in names:
        o=pd.read_parquet(arts[n][1]).drop_duplicates('contact_id').set_index('contact_id')
        v=np.nan_to_num(o['prob'].reindex(td.contact_id.values).values,nan=0.0); Rt[n]=rankdata(v)/len(v)
    def tblend(w):
        s=np.zeros(len(td))
        for n,c in w.items(): s+=Rt[n]*c
        return s/sum(w.values())
    isgt=(td.p2=='G').values
    stg=tblend(wg); stp=tblend(wp)
    def apply(scores,mask,w,keys,q):
        d=td[mask].copy(); d['prob']=scores[mask]
        d=smooth_by(d,w,keys)
        thr=np.quantile(d.sm.values,1-q)
        return d.contact_id.values, (d.sm.values>=thr).astype(int)
    idg,pg=apply(stg,isgt,wgr,('game_play','p1'),qg)
    idp,pp=apply(stp,~isgt,wpr,('game_play','p1','p2'),qp)
    pred=pd.DataFrame({'contact_id':np.concatenate([idg,idp]),'contact':np.concatenate([pg,pp])})
    out=pred.set_index('contact_id')['contact'].reindex(ss.contact_id).reset_index()
    assert out.contact.notna().all() and len(out)==len(ss)
    out.to_csv(os.path.join(HERE,'sub_final_opt.csv'),index=False)
    print('test posrate %.4f (g %.4f p %.4f)'%(out.contact.mean(),pg.mean(),pp.mean()),flush=True)

if __name__=='__main__': main()
