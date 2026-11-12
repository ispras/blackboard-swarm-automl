import numpy as np, pandas as pd, lightgbm as lgb, json, time
from sklearn.metrics import roc_auc_score

t0=time.time()
def log(*a): print(f'[{time.time()-t0:.1f}s]',*a,flush=True)

R='.'; C='../castor_alpha'; A='../achernar_sigma'

base_models = {
 'p_rigel': (f'{R}/oof_ground.parquet', f'{R}/testpred_ground.parquet', 'prob'),
 'p_ach':   (f'{A}/oof_ground_lgbv1.parquet', f'{A}/testpred_ground_lgbv1.parquet', 'pred'),
 'p_ctab':  (f'{C}/ground_oof_tab.parquet', f'{C}/ground_test_tab.parquet', 'oof/pred'),
 'p_cvid':  (f'{C}/ground_oof_video.parquet', f'{C}/ground_test_video.parquet', 'pred'),
 'p_gfus':  (f'{A}/oof_ground_gfus.parquet', f'{A}/testpred_ground_gfus.parquet', 'pred'),
 'p_ceff':  (f'{C}/ground_oof_video_eff.parquet', f'{C}/ground_test_video_eff.parquet', 'pred'),
 'p_r34':   (f'{C}/ground_oof_video_r34.parquet', f'{C}/ground_test_video_r34.parquet', 'pred'),
 'p_cnx':   (f'{C}/ground_oof_video_cnx.parquet', f'{C}/ground_test_video_cnx.parquet', 'pred'),
 'p_eb3':   (f'{C}/ground_oof_video_eb3.parquet', f'{C}/ground_test_video_eb3.parquet', 'pred'),
 'p_cnx2':  (f'{C}/ground_oof_video_cnx2.parquet', f'{C}/ground_test_video_cnx2.parquet', 'pred'),
 'p_wide':  (f'{C}/ground_oof_video_wide.parquet', f'{C}/ground_test_video_wide.parquet', 'pred'),
 'p_tight': (f'{C}/ground_oof_video_tight.parquet', f'{C}/ground_test_video_tight.parquet', 'pred'),
}

rg = pd.read_parquet(f'{R}/oof_ground.parquet')[['contact_id','game_play','step','nfl_player_id_1','contact','fold']]
rg = rg.sort_values(['game_play','nfl_player_id_1','step']).reset_index(drop=True)
g = rg.rename(columns={'nfl_player_id_1':'pid'})
import os
for name,(op,tp,col) in base_models.items():
    if not (os.path.exists(op) and os.path.exists(tp)):
        log('skip missing base',name); continue
    oc = col.split('/')[0]
    d = pd.read_parquet(op)[['contact_id',oc]].rename(columns={oc:name})
    g = g.merge(d,on='contact_id',how='left')
log('merged base', g.shape, g[[c for c in g.columns if c.startswith('p_')]].isna().mean().to_dict())

cols=[c for c in g.columns if c.startswith('p_')]
# temporal features per player series
g = g.sort_values(['game_play','pid','step']).reset_index(drop=True)
grp = g.groupby(['game_play','pid'], sort=False)
feat={}
for c in cols:
    v=g[c].values
    feat[c]=v
    feat[c+'_lag1']=grp[c].shift(1).values
    feat[c+'_lead1']=grp[c].shift(-1).values
    feat[c+'_lag2']=grp[c].shift(2).values
    feat[c+'_lead2']=grp[c].shift(-2).values
    feat[c+'_lag3']=grp[c].shift(3).values
    feat[c+'_lead3']=grp[c].shift(-3).values
    feat[c+'_m5']=grp[c].transform(lambda x:x.rolling(5,center=True,min_periods=1).mean()).values
    feat[c+'_m11']=grp[c].transform(lambda x:x.rolling(11,center=True,min_periods=1).mean()).values
    feat[c+'_std5']=grp[c].transform(lambda x:x.rolling(5,center=True,min_periods=1).std()).values
X = pd.DataFrame(feat)
X['step']=g['step'].values

ftr = pd.read_parquet(f'{C}/ground_train_feats.parquet')
rawcols=[c for c in ftr.columns if c!='contact_id']
ftr2 = ftr[['contact_id']+rawcols].drop_duplicates('contact_id')
tmp = g[['contact_id']].merge(ftr2,on='contact_id',how='left')
for c in rawcols:
    X[c]=tmp[c].values
X = X.fillna(0)
y=g.contact.values.astype(np.int8); folds=g.fold.values
log('X',X.shape)

# test matrix
te = pd.read_parquet(f'{R}/testpred_ground.parquet')[['contact_id','game_play','step','nfl_player_id_1']]
te = te.rename(columns={'nfl_player_id_1':'pid'})
te = te.sort_values(['game_play','pid','step']).reset_index(drop=True)
for name,(op,tp,col) in base_models.items():
    if not (os.path.exists(op) and os.path.exists(tp)):
        continue
    tcol = col.split('/')[-1]
    d = pd.read_parquet(tp)[['contact_id',tcol]].rename(columns={tcol:name})
    te = te.merge(d,on='contact_id',how='left')
te = te.sort_values(['game_play','pid','step']).reset_index(drop=True)
grpt = te.groupby(['game_play','pid'], sort=False)
feat={}
for c in cols:
    feat[c]=te[c].values
    feat[c+'_lag1']=grpt[c].shift(1).values
    feat[c+'_lead1']=grpt[c].shift(-1).values
    feat[c+'_lag2']=grpt[c].shift(2).values
    feat[c+'_lead2']=grpt[c].shift(-2).values
    feat[c+'_lag3']=grpt[c].shift(3).values
    feat[c+'_lead3']=grpt[c].shift(-3).values
    feat[c+'_m5']=grpt[c].transform(lambda x:x.rolling(5,center=True,min_periods=1).mean()).values
    feat[c+'_m11']=grpt[c].transform(lambda x:x.rolling(11,center=True,min_periods=1).mean()).values
    feat[c+'_std5']=grpt[c].transform(lambda x:x.rolling(5,center=True,min_periods=1).std()).values
Xt = pd.DataFrame(feat); Xt['step']=te['step'].values
fte = pd.read_parquet(f'{C}/ground_test_feats.parquet')
fte2 = fte[['contact_id']+rawcols].drop_duplicates('contact_id')
tmp = te[['contact_id']].merge(fte2,on='contact_id',how='left')
for c in rawcols: Xt[c]=tmp[c].values
Xt = Xt.fillna(0)
Xt = Xt[X.columns]
log('Xt',Xt.shape)

params=dict(objective='binary',metric='auc',learning_rate=0.03,num_leaves=63,min_child_samples=80,
            subsample=0.8,colsample_bytree=0.7,reg_lambda=1.0,n_jobs=10,verbose=-1,seed=42)
oof=np.zeros(len(g)); test_pred=np.zeros(len(te)); rounds=[]
for f in sorted(np.unique(folds)):
    tr=folds!=f; va=folds==f
    dtr=lgb.Dataset(X[tr],label=y[tr]); dva=lgb.Dataset(X[va],label=y[va],reference=dtr)
    m=lgb.train(params,dtr,num_boost_round=2500,valid_sets=[dva],callbacks=[lgb.early_stopping(100,verbose=False)])
    oof[va]=m.predict(X[va],num_iteration=m.best_iteration)
    test_pred+=m.predict(Xt,num_iteration=m.best_iteration)/len(np.unique(folds))
    rounds.append(m.best_iteration)
    log('fold',f,'iter',m.best_iteration,'auc',round(roc_auc_score(y[va],oof[va]),5))

from postprocess_search import smooth
out=pd.DataFrame({'contact_id':g.contact_id,'game_play':g.game_play,'nfl_player_id_1':g.pid,
                  'nfl_player_id_2':-1,'step':g.step,'contact':y,'fold':folds,'prob':oof})
out.to_parquet('ground_oof_stack_r.parquet',index=False)
tout=pd.DataFrame({'contact_id':te.contact_id,'game_play':te.game_play,'nfl_player_id_1':te.pid,
                   'nfl_player_id_2':-1,'step':te.step,'prob':test_pred.astype(np.float32)})
tout.to_parquet('ground_test_stack_r.parquet',index=False)

best=(-1,None,None)
for w in [1,3,5,7,9,11,15,21]:
    if w==1: p=oof.astype(np.float64)
    else:
        ds,p=smooth(out,oof,w); p=p.astype(np.float64)
    # mcc
    idx=np.argsort(-p); ys=y[idx].astype(np.float64); tp=np.cumsum(ys); fp=np.cumsum(1-ys)
    pos=ys.sum(); neg=len(ys)-pos; fn=pos-tp; tn=neg-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); mm=np.where(den>0,(tp*tn-fp*fn)/den,-1)
    k=int(np.argmax(mm)); r=(float(mm[k]),float(np.sort(p)[::-1][k]))
    log('smooth',w,r)
    if r[0]>best[0]: best=(r[0],w,r[1])
log('BEST',best,'rounds',rounds,'auc',roc_auc_score(y,oof))
json.dump({'mcc':best[0],'window':best[1],'thr':best[2],'rounds':rounds,'auc':float(roc_auc_score(y,oof))},open('ground_stack_r.json','w'),indent=2)
