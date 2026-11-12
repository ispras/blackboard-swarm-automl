import os, gc, time, sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef, roc_auc_score

W='workspaces/capella_pi'
t0=time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]',*a,flush=True)

tr=pd.read_parquet(f'{W}/train_feat.parquet')
te=pd.read_parquet(f'{W}/test_feat.parquet')
# reconstruct contact_id
def mkid(d):
    return (d['game_play'].astype(str)+'_'+d['step'].astype(str)+'_'+
            d['nfl_player_id_1'].astype(str)+'_'+d['nfl_player_id_2'].astype(str))
tr['contact_id']=mkid(tr); te['contact_id']=mkid(te)

folds=pd.read_csv('workspaces/alkaid_psi/cv_folds.csv')
tr=tr.merge(folds,on='game_play',how='left')
assert tr.fold.notna().all(), tr.fold.isna().sum()

drop=set(['contact','contact_id','game_play','nfl_player_id_1','nfl_player_id_2','fold'])
for c in list(tr.columns):
    if c.endswith('is_test') or c=='is_test' or c in ('p1key','p2key','p2id'):
        drop.add(c)
feats=[c for c in tr.columns if c not in drop]
# object -> categorical codes. p1_team,p1_position,p2_team,p2_position are category dtype
cat_feats=['p1_team','p1_position','p2_team','p2_position']
for c in cat_feats:
    tr[c]=tr[c].astype('category')
    te[c]=te[c].astype('category')
    # align categories
    cats=tr[c].cat.categories.union(te[c].cat.categories)
    tr[c]=tr[c].cat.set_categories(cats)
    te[c]=te[c].cat.set_categories(cats)
# ensure numeric
for c in feats:
    if c not in cat_feats and tr[c].dtype==object:
        tr[c]=pd.to_numeric(tr[c],errors='coerce')
        te[c]=pd.to_numeric(te[c],errors='coerce')
Y=tr['contact'].values.astype(np.int8)
X=tr[feats]
log('data',X.shape,'pos',Y.mean())

params=dict(objective='binary',metric='auc',learning_rate=0.06,num_leaves=96,
            min_data_in_leaf=100,feature_fraction=0.7,bagging_fraction=0.8,
            bagging_freq=1,lambda_l2=5.0,num_threads=12,verbosity=-1,
            max_bin=127,seed=42)
NEG_FRAC=0.20

oof=np.zeros(len(tr),dtype=np.float32)
test_pred=np.zeros(len(te),dtype=np.float32)
best_iters=[]
NF=5
for f in range(NF):
    va=np.where(tr.fold.values==f)[0]
    tri_all=np.where(tr.fold.values!=f)[0]
    rng=np.random.RandomState(100+f)
    neg=tri_all[Y[tri_all]==0]; pos=tri_all[Y[tri_all]==1]
    negs=rng.choice(neg,int(len(neg)*NEG_FRAC),replace=False)
    tri=np.sort(np.concatenate([pos,negs]))
    dtr=lgb.Dataset(X.iloc[tri],label=Y[tri],categorical_feature=cat_feats,free_raw_data=True)
    dva=lgb.Dataset(X.iloc[va],label=Y[va],categorical_feature=cat_feats,reference=dtr,free_raw_data=True)
    m=lgb.train(params,dtr,num_boost_round=2500,valid_sets=[dva],
                callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(200)])
    best_iters.append(m.best_iteration)
    oof[va]=m.predict(X.iloc[va],num_iteration=m.best_iteration).astype(np.float32)
    test_pred += m.predict(te[feats],num_iteration=m.best_iteration).astype(np.float32)/NF
    auc=roc_auc_score(Y[va],oof[va]); log('fold',f,'best_iter',m.best_iteration,'auc',round(auc,5))
    del dtr,dva,m; gc.collect()

test_pred=test_pred.astype(np.float32)
auc_all=roc_auc_score(Y,oof)
# threshold sweep
best_mcc=-1; best_th=0.5
ths=np.unique(np.quantile(oof,np.linspace(0.5,0.9999,600)))
# coarse then fine around best
for th in ths:
    mcc=matthews_corrcoef(Y,oof>th)
    if mcc>best_mcc: best_mcc,best_th=mcc,th
log('OOF AUC',round(auc_all,5),'best MCC',round(best_mcc,5),'th',best_th,'best_iters',best_iters)

pd.DataFrame({'contact_id':tr.contact_id,'pred':oof,'fold':tr.fold.values}).to_parquet(f'{W}/oof_lgbm.parquet',index=False)
pd.DataFrame({'contact_id':te.contact_id,'pred':test_pred}).to_parquet(f'{W}/test_lgbm.parquet',index=False)
json.dump({'auc':float(auc_all),'mcc':float(best_mcc),'th':float(best_th),
           'best_iters':best_iters,'n_feats':len(feats),'feats':feats},
          open(f'{W}/lgbm_meta.json','w'),indent=1)
log('saved artifacts')
