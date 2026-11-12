import os, sys, time, json, gc
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold

HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')
DATA=os.path.join(HERE,'..','..','data')

def best_mcc(y,p):
    order=np.argsort(-p); ys=y[order].astype(np.float64); ps=p[order]
    tp=np.cumsum(ys); fp=np.cumsum(1.0-ys)
    P=float(y.sum()); N=float(len(y))-P; fn=P-tp; tn=N-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc=np.where(den>0,(tp*tn-fp*fn)/np.where(den>0,den,1.0),0.0)
    i=int(np.argmax(mcc)); return float(mcc[i]), float((ps[i]+ps[min(i+1,len(ps)-1)])/2)

def smooth(tmp,w,keys=('game_play','p1','p2')):
    r=tmp.groupby(list(keys),sort=False)['prob'].rolling(w,min_periods=1,center=True).mean()
    return r.reset_index(level=[0,1,2],drop=True).reindex(tmp.index).values

def parse(df):
    parts=df.contact_id.str.split('_')
    df=df.copy()
    df['game_play']=parts.str[0]+'_'+parts.str[1]; df['step']=parts.str[2].astype(int)
    df['p1']=parts.str[3]; df['p2']=parts.str[4]
    return df

def xpair_map(base):
    # base: contact_id,prob,game_play,step,p1,p2
    pair=base[base.p2!='G'][['game_play','step','p1','p2','prob']].copy()
    a=pair.rename(columns={'p1':'player'})[['game_play','step','player','prob']]
    b=pair.rename(columns={'p2':'player'})[['game_play','step','player','prob']]
    long=pd.concat([a,b],ignore_index=True)
    g=long.groupby(['game_play','step','player'])['prob'].agg(xpair_max='max',xpair_mean='mean').reset_index()
    return g

def build(base, ctx):
    df=base.merge(ctx,on='contact_id',how='left')
    # temporal on base prob per pair
    df=df.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    g=df.groupby(['game_play','p1','p2'],sort=False)
    df['base']=df.prob
    for l in (1,2):
        df[f'base_prev{l}']=g['base'].shift(l)
        df[f'base_next{l}']=g['base'].shift(-l)
    df['base_roll3max']=g['base'].transform(lambda s:s.rolling(3,min_periods=1,center=True).max())
    df['base_roll5mean']=g['base'].transform(lambda s:s.rolling(5,min_periods=1,center=True).mean())
    if 'cnn' in df.columns:
        df['cnn_prev1']=g['cnn'].shift(1)
        df['cnn_next1']=g['cnn'].shift(-1)
        df['cnn_roll3mean']=g['cnn'].transform(lambda s:s.rolling(3,min_periods=1,center=True).mean())
        df['cnn_roll5mean']=g['cnn'].transform(lambda s:s.rolling(5,min_periods=1,center=True).mean())
    return df

def main():
    t0=time.time()
    # ground model oof/test for extra signal
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    base=parse(pd.read_parquet(os.path.join(HERE,'oof_pred.parquet'))).merge(lab,on='contact_id',how='left')
    # sirius ground-CNN global prob as extra feature
    sc=pd.read_parquet(os.path.join(HERE,'..','sirius_tau','oof_pred.parquet')).rename(columns={'prob':'cnn'})
    base=base.merge(sc,on='contact_id',how='left')
    # ctx features (now with relative helmet-pose features)
    rp=pd.read_parquet(os.path.join(CACHE,'relpose_train.parquet'))
    relc=[c for c in rp.columns if c not in ('game_play','step','player')]
    ctx_cols=['contact_id','nn_dist1','cnt21','cnt51','speed1','dist','dmin_w5'] \
             +[c+'_p1' for c in relc]+[c+'_p2' for c in relc]
    ctx=pd.read_parquet(os.path.join(CACHE,'featg_rel_all_train.parquet'),columns=ctx_cols)
    df=build(base,ctx)
    # xpair per player
    xm=xpair_map(base)
    # ground rows use xpair of p1, also pair rows use xpair of p1 (own player context)
    df=df.merge(xm.rename(columns={'player':'p1'}),on=['game_play','step','p1'],how='left')
    df=df.merge(xm.rename(columns={'player':'p2','xpair_max':'xpair_max_p2','xpair_mean':'xpair_mean_p2'}),
                on=['game_play','step','p2'],how='left')
    # neighbor xpair for p1
    df=df.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    g=df.groupby(['game_play','p1','p2'],sort=False)
    df['xp_prev1']=g['xpair_max'].shift(1); df['xp_prev2']=g['xpair_max'].shift(2)
    df['xp_next1']=g['xpair_max'].shift(-1)
    df['is_ground']=(df.p2=='G').astype(np.int8)
    feats=['base','base_prev1','base_prev2','base_next1','base_next2','base_roll3max','base_roll5mean',
           'cnn','cnn_prev1','cnn_next1','cnn_roll3mean','cnn_roll5mean',
           'step','is_ground','nn_dist1','cnt21','cnt51','speed1','dist','dmin_w5',
           'xpair_max','xpair_mean','xp_prev1','xp_prev2','xp_next1','xpair_max_p2','xpair_mean_p2'] \
          +[c+'_p1' for c in relc]+[c+'_p2' for c in relc]
    y=df.contact.values.astype(np.int8); groups=df.game_play.values
    params=dict(objective='binary',learning_rate=0.03,num_leaves=64,min_data_in_leaf=100,
                feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,
                n_jobs=16,seed=11,verbose=-1,metric='auc')
    oof=np.zeros(len(df)); iters=[]
    gkf=GroupKFold(n_splits=5)
    for k,(tri,vai) in enumerate(gkf.split(df,y,groups)):
        dtr=lgb.Dataset(df.iloc[tri][feats],y[tri],categorical_feature=['is_ground'])
        dva=lgb.Dataset(df.iloc[vai][feats],y[vai],categorical_feature=['is_ground'],reference=dtr)
        m=lgb.train(params,dtr,num_boost_round=2000,valid_sets=[dva],
                    callbacks=[lgb.early_stopping(80,verbose=False)])
        oof[vai]=m.predict(df.iloc[vai][feats],num_iteration=m.best_iteration)
        iters.append(m.best_iteration); m.save_model(os.path.join(CACHE,f'm_l2_{k}.txt'))
        print('fold %d iter %d t=%d'%(k,m.best_iteration,time.time()-t0),flush=True)
        del dtr,dva,m; gc.collect()
    raw,thr=best_mcc(y,oof)
    tmp=df[['game_play','p1','p2','step','contact']].copy(); tmp['prob']=oof
    tmp=tmp.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    best=(-1,0,0)
    for w in (1,3,5,7,9):
        pp=smooth(tmp,w); mm,tt=best_mcc(tmp.contact.values,pp)
        print('smooth w%d %.5f thr %.4f'%(w,mm,tt),flush=True)
        if mm>best[0]: best=(mm,w,tt)
    print('L2 raw %.5f thr %.4f | BEST smooth %.5f w%d thr %.4f t=%d'%(raw,thr,best[0],best[1],best[2],time.time()-t0),flush=True)
    np.save(os.path.join(CACHE,'oof_l2.npy'),oof.astype(np.float32))
    np.save(os.path.join(CACHE,'ids_l2.npy'),df.contact_id.values)
    json.dump(dict(raw=raw,thr=thr,smooth=best[0],w=best[1],sthr=best[2],iters=iters,feats=feats),
              open(os.path.join(CACHE,'res_l2.json'),'w'))
    # test
    tb=parse(pd.read_parquet(os.path.join(HERE,'test_pred.parquet')))
    sct=pd.read_parquet(os.path.join(HERE,'..','sirius_tau','test_pred.parquet')).rename(columns={'prob':'cnn'})
    tb=tb.merge(sct,on='contact_id',how='left')
    tctx=pd.read_parquet(os.path.join(CACHE,'featg_rel_all_test.parquet'),columns=ctx_cols)
    tdf=build(tb,tctx)
    txm=xpair_map(tb)
    tdf=tdf.merge(txm.rename(columns={'player':'p1'}),on=['game_play','step','p1'],how='left')
    tdf=tdf.merge(txm.rename(columns={'player':'p2','xpair_max':'xpair_max_p2','xpair_mean':'xpair_mean_p2'}),
                  on=['game_play','step','p2'],how='left')
    tdf=tdf.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    tg=tdf.groupby(['game_play','p1','p2'],sort=False)
    tdf['xp_prev1']=tg['xpair_max'].shift(1); tdf['xp_prev2']=tg['xpair_max'].shift(2)
    tdf['xp_next1']=tg['xpair_max'].shift(-1)
    tdf['is_ground']=(tdf.p2=='G').astype(np.int8)
    pt=np.zeros(len(tdf))
    for k in range(5):
        m=lgb.Booster(model_file=os.path.join(CACHE,f'm_l2_{k}.txt'))
        pt+=m.predict(tdf[feats],num_iteration=iters[k])/5
    np.save(os.path.join(CACHE,'ptest_l2.npy'),pt.astype(np.float32))
    np.save(os.path.join(CACHE,'pids_l2.npy'),tdf.contact_id.values)
    pd.DataFrame({'contact_id':df.contact_id.values,'prob':oof}).to_parquet(os.path.join(HERE,'oof_pred_l2.parquet'))
    pd.DataFrame({'contact_id':tdf.contact_id.values,'prob':pt}).to_parquet(os.path.join(HERE,'test_pred_l2.parquet'))
    print('saved test',len(tdf),flush=True)

if __name__=='__main__':
    main()
