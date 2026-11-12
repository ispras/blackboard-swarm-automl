import os, sys, glob, json
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

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

def main():
    arts=discover(); cn=[n for n in arts if n.startswith('sirius')][0]
    lab=pd.read_csv(os.path.join(DATA,'train_labels.csv'),usecols=['contact_id','contact'])
    ref=pd.read_parquet(os.path.join(HERE,'oof_pred.parquet'),columns=['contact_id']).merge(lab,on='contact_id',how='left')
    ref['game_play'],ref['step'],ref['p1'],ref['p2']=keys_of(ref.contact_id)
    ref=ref.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    isg=(ref.p2=='G').values
    # model ranks over all, keep ground
    df=pd.DataFrame({'contact_id':ref.contact_id,'game_play':ref.game_play,'step':ref.step,'p1':ref.p1,'p2':ref.p2,'contact':ref.contact})
    for n,(op,tp) in arts.items():
        o=pd.read_parquet(op).drop_duplicates('contact_id').set_index('contact_id')
        v=np.nan_to_num(o['prob'].reindex(ref.contact_id.values).values,nan=0.0)
        df[n]=rankdata(v)/len(v)
    # rel pose p1
    rp=pd.read_parquet(os.path.join(CACHE,'relpose_train.parquet'))
    relc=[c for c in rp.columns if c not in ('game_play','step','player')]
    rp1=rp.rename(columns={'player':'p1',**{c:c+'_r1' for c in relc}})
    df['_p1n']=df.p1.astype(np.int64); rp1['_p1n']=rp1.p1.astype(np.int64)
    df=df.merge(rp1.drop(columns=['p1']),on=['game_play','step','_p1n'],how='left')
    # context
    ctx=pd.read_parquet(os.path.join(CACHE,'featg_rel_all_train.parquet'),columns=['contact_id','nn_dist1','cnt21','cnt51','speed1','speed2','acceleration1','dist','dmin_w5'])
    df=df.merge(ctx,on='contact_id',how='left')
    # temporal for key model cols
    keycols=['procyon_theta_l2','procyon_theta_gallr','procyon_theta_gruc',cn,'procyon_theta_seqc']
    keycols=[c for c in keycols if c in df.columns]
    df=df.sort_values(['game_play','p1','step']).reset_index(drop=True)
    g=df.groupby(['game_play','p1'],sort=False)
    for c in keycols:
        df[c+'_prev']=g[c].shift(1); df[c+'_next']=g[c].shift(-1)
    feats=[c for c in df.columns if c not in ('contact_id','game_play','step','p1','p2','contact','_p1n') and df[c].dtype!=object]
    dg=df[df.p2=='G'].reset_index(drop=True)
    y=dg.contact.values.astype(np.int8); groups=dg.game_play.values
    params=dict(objective='binary',learning_rate=0.03,num_leaves=64,min_data_in_leaf=60,
                feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=2.0,n_jobs=16,seed=21,verbose=-1,metric='auc')
    oof=np.zeros(len(dg)); iters=[]
    for k,(tri,vai) in enumerate(GroupKFold(n_splits=5).split(dg,y,groups)):
        dtr=lgb.Dataset(dg.iloc[tri][feats],y[tri]); dva=lgb.Dataset(dg.iloc[vai][feats],y[vai],reference=dtr)
        m=lgb.train(params,dtr,num_boost_round=1500,valid_sets=[dva],callbacks=[lgb.early_stopping(80,verbose=False)])
        oof[vai]=m.predict(dg.iloc[vai][feats],num_iteration=m.best_iteration); iters.append(m.best_iteration)
        m.save_model(os.path.join(CACHE,f'm_l3g_{k}.txt'))
    mcc,thr=best_mcc(y,oof)
    # smooth
    tmp=dg[['game_play','p1','step','contact']].copy(); tmp['prob']=oof
    tmp=tmp.sort_values(['game_play','p1','step']).reset_index(drop=True)
    grp=tmp.groupby(['game_play','p1'],sort=False).ngroup().values
    best=(-1,0,0)
    for w in (1,3,5):
        sm=pd.Series(tmp.prob.values).groupby(grp).rolling(w,min_periods=1,center=True).mean().values
        mm,tt=best_mcc(tmp.contact.values,sm)
        print('ground-stack smooth w%d %.5f'%(w,mm),flush=True)
        if mm>best[0]: best=(mm,w,tt)
    print('GROUND STACK raw %.5f thr %.4f | best %.5f w%d thr %.5f'%(mcc,thr,best[0],best[1],best[2]),flush=True)
    np.save(os.path.join(CACHE,'oof_l3g.npy'),oof.astype(np.float32)); np.save(os.path.join(CACHE,'ids_l3g.npy'),dg.contact_id.values)
    json.dump(dict(raw=mcc,thr=thr,smooth=best[0],w=best[1],sthr=best[2],iters=iters,feats=feats),open(os.path.join(CACHE,'res_l3g.json'),'w'))
    # test
    sst=pd.read_csv(os.path.join(DATA,'sample_submission.csv'))
    td=pd.DataFrame({'contact_id':sst.contact_id}); td['game_play'],td['step'],td['p1'],td['p2']=keys_of(td.contact_id)
    td=td.sort_values(['game_play','p1','p2','step']).reset_index(drop=True)
    for n,(op,tp) in arts.items():
        o=pd.read_parquet(tp).drop_duplicates('contact_id').set_index('contact_id')
        v=np.nan_to_num(o['prob'].reindex(td.contact_id.values).values,nan=0.0)
        td[n]=rankdata(v)/len(v)
    rp1t=rp.copy(); rp1t['_p1n']=rp1t.player.astype(np.int64)
    rp1t=rp1t.rename(columns={**{c:c+'_r1' for c in relc}})
    td['_p1n']=td.p1.astype(np.int64)
    td=td.merge(rp1t.drop(columns=['player']),on=['game_play','step','_p1n'],how='left')
    ctxt=pd.read_parquet(os.path.join(CACHE,'featg_rel_all_test.parquet'),columns=['contact_id','nn_dist1','cnt21','cnt51','speed1','speed2','acceleration1','dist','dmin_w5'])
    td=td.merge(ctxt,on='contact_id',how='left')
    td=td.sort_values(['game_play','p1','step']).reset_index(drop=True)
    gt=td.groupby(['game_play','p1'],sort=False)
    for c in keycols:
        td[c+'_prev']=gt[c].shift(1); td[c+'_next']=gt[c].shift(-1)
    dgt=td[(td.p2=='G')].reset_index(drop=True)
    pt=np.zeros(len(dgt))
    for k in range(5):
        m=lgb.Booster(model_file=os.path.join(CACHE,f'm_l3g_{k}.txt'))
        pt+=m.predict(dgt[feats],num_iteration=iters[k])/5
    np.save(os.path.join(CACHE,'ptest_l3g.npy'),pt.astype(np.float32)); np.save(os.path.join(CACHE,'pids_l3g.npy'),dgt.contact_id.values)
    print('saved ground test',len(dgt),flush=True)

if __name__=='__main__': main()
