import os, sys, json
import numpy as np, pandas as pd, lightgbm as lgb

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')
DATA=os.path.join(HERE,'..','..','data')

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def smooth_vals(tmp,w):
    r=tmp.groupby(['game_play','p1','p2'],sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    return r.reset_index(level=[0,1,2],drop=True).reindex(tmp.index).values

def parse(cid):
    parts=cid.str.split('_')
    return parts.str[0]+'_'+parts.str[1], parts.str[2].astype(int), parts.str[3], parts.str[4]

def main():
    tag=sys.argv[1] if len(sys.argv)>1 else 'gall'
    ids=np.load(os.path.join(CACHE,f'ids_{tag}.npy'),allow_pickle=True)
    oof=np.load(os.path.join(CACHE,f'oof_{tag}.npy')).astype(np.float64)
    gallery_folds=5
    # test preds
    dt=pd.read_parquet(os.path.join(CACHE,'featg_all_test.parquet'))
    feats=lgb.Booster(model_file=os.path.join(CACHE,f'm_{tag}_0.txt')).feature_name()
    pt=np.zeros(len(dt))
    for k in range(gallery_folds):
        m=lgb.Booster(model_file=os.path.join(CACHE,f'm_{tag}_{k}.txt'))
        pt+=m.predict(dt[feats])/gallery_folds
    pd.DataFrame({'contact_id':dt.contact_id.values,'prob':pt}).to_parquet(os.path.join(HERE,'test_pred.parquet'))
    pd.DataFrame({'contact_id':ids,'prob':oof}).to_parquet(os.path.join(HERE,'oof_pred.parquet'))
    print('wrote artifacts', len(ids), len(dt))
    # eval
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    m=pd.DataFrame({'contact_id':ids,'prob':oof}).merge(lab,on='contact_id',how='left')
    gp,step,p1,p2=parse(m.contact_id)
    m['game_play']=gp; m['p1']=p1; m['p2']=p2; m['step']=step
    m['is_ground']=(m.p2=='G')
    for name,msk in [('all',np.ones(len(m),bool)),('ground',m.is_ground.values),('pair',(~m.is_ground).values)]:
        mm,tt=best_mcc(m.contact.values[msk],oof[msk]); print(name,'n',msk.sum(),'raw MCC %.5f thr %.4f'%(mm,tt))
    # blend with ground model on ground rows
    gi=np.load(os.path.join(CACHE,'ids_train_ground.npy'),allow_pickle=True)
    go=np.load(os.path.join(CACHE,'oof_train_ground.npy')).astype(np.float64)
    gmap=dict(zip(gi,go))
    m['gprob']=m.contact_id.map(gmap)
    print('ground rows matched', m.gprob.notna().sum(), 'of', m.is_ground.sum())
    # rank transform
    from scipy.stats import rankdata
    tmp=m[['game_play','p1','p2','step','contact','prob','gprob','is_ground']].copy()
    for a in [0.0,0.2,0.3,0.4,0.5]:
        blend=tmp.prob.values.copy()
        msk=tmp.is_ground.values
        r_all=rankdata(tmp.prob.values,method='average')/len(tmp)
        r_g=rankdata(np.where(np.isnan(tmp.gprob.values),-1,tmp.gprob.values),method='average')/len(tmp)
        blend=np.where(msk,(1-a)*r_all+a*r_g,r_all)
        t2=tmp.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
        b2=pd.Series(blend).iloc[tmp.sort_values(['game_play','p1','p2','step']).index].values
        t2['prob']=b2
        best=(-1,0)
        for w in (5,7,9):
            pp=smooth_vals(t2,w); mm,tt=best_mcc(t2.contact.values,pp)
            if mm>best[0]: best=(mm,w)
        print('alpha_ground %.1f -> best smooth MCC %.5f w%d'%(a,best[0],best[1]))

if __name__=='__main__':
    main()
