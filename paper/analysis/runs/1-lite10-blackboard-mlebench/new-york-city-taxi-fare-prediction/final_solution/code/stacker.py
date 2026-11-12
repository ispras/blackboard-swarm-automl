import sys, os, json, glob, argparse
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0,'/download_job_code/payload/sandbox/workspaces/aldebaran_omicron')
from blend import load_models, find_test
ROOT='/download_job_code/payload/sandbox'
ME=f'{ROOT}/workspaces/aldebaran_omicron'; MIM=f'{ROOT}/workspaces/mimosa_omicron'
NTEST=9914

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='stack'); a=ap.parse_args()
    M=load_models()
    names=[n for n,d in M.items() if d['test'] is not None and len(d['test'])==NTEST]
    print('models',names)
    common=None
    for n in names:
        s=set(M[n]['orig'].tolist()); common=s if common is None else common&s
    common=np.array(sorted(common),dtype=np.int64)
    oref=np.load(f'{ME}/holdout_origidx.npy'); yref=np.load(f'{ME}/holdout_y.npy')
    ymap=dict(zip(oref.tolist(),yref.tolist()))
    common=np.array([int(i) for i in common if int(i) in ymap],dtype=np.int64)
    y=np.array([ymap[int(i)] for i in common],dtype=np.float32)
    B=[]
    for n in names:
        d=M[n]; s=pd.Series(np.arange(len(d['orig'])),index=d['orig'])
        B.append(d['pred'][s.loc[common].to_numpy()].astype(np.float32))
    B=np.vstack(B).T
    # features
    Xall=np.load(f'{ME}/shared_X.npy',mmap_mode='r'); oi=np.load(f'{ME}/shared_origidx.npy',mmap_mode='r')
    fn=list(np.load(f'{ME}/shared_featnames.npy'))
    pos=np.searchsorted(np.asarray(oi),common)
    SX=np.asarray(Xall[pos])
    featsel=['dist','dist_man','year','month','hour','weekday','passenger_count','p_airport_min',
             'd_airport_min','te_route_oof','route_count','p_cell_count','d_cell_count','bearing','p_mid','d_mid']
    idx=[fn.index(f) for f in featsel]
    F=np.column_stack([SX[:,i] for i in idx])
    X=np.column_stack([B,F])
    cols=[f'pred_{n}' for n in names]+featsel
    print('stack matrix',X.shape)
    from sklearn.model_selection import KFold
    params=dict(objective='regression',metric='rmse',learning_rate=0.03,num_leaves=63,min_data_in_leaf=200,
                feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,max_bin=255,num_threads=12,verbose=-1,seed=0)
    kf=KFold(5,shuffle=True,random_state=0); oof=np.zeros(len(y))
    for tr,te in kf.split(X):
        dtr=lgb.Dataset(X[tr],y[tr],feature_name=cols); dva=lgb.Dataset(X[te],y[te],reference=dtr)
        m=lgb.train(params,dtr,num_boost_round=2000,valid_sets=[dva],callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
        oof[te]=m.predict(X[te],num_iteration=m.best_iteration)
    cv=float(np.sqrt(np.mean((oof-y)**2)))
    print('STACK_CV_RMSE',cv)
    # fit on all
    dtr=lgb.Dataset(X,y,feature_name=cols)
    m=lgb.train(params,dtr,num_boost_round=800)
    # test
    sys.path.insert(0,f'{MIM}')
    from test_features import build_test, add_counts_te
    from common import load_te
    df=build_test(); df=add_counts_te(df,load_te()); corrupt=df['corrupt'].to_numpy().astype(bool)
    Bt=np.vstack([M[n]['test'] for n in names]).T
    Ft=df[featsel].to_numpy(np.float32)
    Xt=np.column_stack([Bt,Ft])
    pv=m.predict(Xt).astype(np.float32)
    # corrupt: use trained corrupt model
    cp=np.load(f'{ME}/corrupt_test_pred.npy')
    pv[corrupt]=cp[corrupt]
    pv=np.clip(pv,2.5,500.0)
    np.save(f'{ME}/test_pred_{a.out}.npy',pv)
    pd.DataFrame({'key':pd.read_csv(f'{ROOT}/data/test.csv')['key'],'fare_amount':pv}).to_csv(f'{ME}/submission_{a.out}.csv',index=False)
    with open(f'{ME}/stack_result.json','w') as f:
        json.dump({'cv_rmse':cv,'models':names,'features':featsel,'valid_mean':float(pv[~corrupt].mean())},f,indent=2)
    print('saved',a.out,'valid mean',pv[~corrupt].mean())

if __name__=='__main__': main()
