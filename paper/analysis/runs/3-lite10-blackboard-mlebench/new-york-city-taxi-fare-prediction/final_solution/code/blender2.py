import pandas as pd, numpy as np, json, os, sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import lightgbm as lgb
W="/download_job_code/payload/sandbox/workspaces"; OUT=f"{W}/alkaid_phi"; DATA="/download_job_code/payload/sandbox/data"
def norm(k):
    k=str(k)
    if '.' not in k: return k
    a,b=k.split('.',1); return a+'.'+b.ljust(9,'0')[:9]
MODELS=[
 ('alkaid_v2lgb', f"{OUT}/v2lgb_val.parquet", f"{OUT}/v2lgb_submission.csv", 'fare_amount'),
 ('antares_xgb_v2', f"{W}/antares_omicron/runs/xgb_v2_val.parquet", f"{W}/antares_omicron/runs/xgb_v2_test.parquet",'pred'),
 ('arcturus_lgb20m', f"{W}/arcturus_upsilon/runs/lgb20m_1023_val.parquet", f"{W}/arcturus_upsilon/runs/lgb20m_1023_test.parquet",'fare_amount'),
 ('arcturus_lgb30m', f"{W}/arcturus_upsilon/runs/lgb30m_val.parquet", f"{W}/arcturus_upsilon/runs/lgb30m_test.parquet",'pred'),
 ('arcturus_te30m', f"{W}/arcturus_upsilon/runs/te30m_val.parquet", f"{W}/arcturus_upsilon/runs/te30m_test.parquet",'pred'),
 ('antares_xgb_te', f"{W}/antares_omicron/runs/xgb_te_val.parquet", f"{W}/antares_omicron/runs/xgb_te_test.parquet",'pred'),
 ('antares_xgb_v3', f"{W}/antares_omicron/runs/xgb_v3_val.parquet", f"{W}/antares_omicron/runs/xgb_v3_test.parquet",'pred'),
 ('antares_cat', f"{W}/antares_omicron/runs/cat_val.parquet", f"{W}/antares_omicron/runs/cat_test.parquet",'pred'),
]
def load_val(path):
    d=pd.read_parquet(path); ycol='y' if 'y' in d.columns else 'fare_amount'
    d=d.rename(columns={ycol:'y'}); d['kn']=d['key'].map(norm)
    return d[['kn','y','pred']]
def load_test(path):
    d=pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
    p=d['pred'].values if 'pred' in d.columns else d['fare_amount'].values
    return pd.DataFrame({'kn':d['key'].map(norm).values,'pred':p}).drop_duplicates('kn')
meta=pd.read_parquet(f"{OUT}/val_meta.parquet"); meta['kn']=meta.key.map(norm)
df=meta[['kn','fare_amount','has_pickup','has_dropoff','distance_km']].copy()
used=[]
for name,vp,tp,_ in MODELS:
    if os.path.exists(vp) and os.path.exists(tp):
        v=load_val(vp)
        df=df.merge(v[['kn','pred']].rename(columns={'pred':name}),on='kn',how='inner'); used.append(name)
    else: print('skip',name)
df=df.dropna(subset=used).reset_index(drop=True)
df['lgd']=np.log1p(df.distance_km.clip(lower=0).fillna(0))
df['year']=df.kn.str[:4].astype(float)
df['missing']=((df.has_pickup==0)|(df.has_dropoff==0))
df['degen']=(~df.missing)&(df.distance_km<0.1)
df['clean']=(~df.missing)&(~df.degen)
y=df.fare_amount.values
print('models',used,'rows',len(df),flush=True)
meta_cols=['lgd','year']
def feats_lin(g): return used
def feats_lgb(g): return used+meta_cols
groups={g:np.where(df[g].values)[0] for g in ['clean','degen','missing']}
kf=KFold(5,shuffle=True,random_state=0)
def run_oof(method):
    oof=np.zeros(len(df))
    for g,idx in groups.items():
        A=df.loc[idx,used].values; yy=df.loc[idx,'fare_amount'].values
        Xm=df.loc[idx,used+meta_cols].values
        for tr,va in kf.split(idx):
            if method=='linear':
                lr=LinearRegression(positive=True).fit(A[tr],yy[tr]); oof[idx[va]]=lr.predict(A[va])
            else:
                ds=lgb.Dataset(Xm[tr],yy[tr])
                p=dict(objective='regression',metric='rmse',learning_rate=0.05,num_leaves=15,
                       min_data_in_leaf=300,feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=1,
                       lambda_l2=5.0,num_threads=16,verbosity=-1,seed=1)
                m=lgb.train(p,ds,num_boost_round=250)
                oof[idx[va]]=m.predict(Xm[va])
    oof=np.clip(oof,2.5,200)
    return oof
oofs={m:run_oof(m) for m in ['linear','lgb']}
for m,o in oofs.items():
    print('OOF',m,'overall',np.sqrt(((o-y)**2).mean()),{g:float(np.sqrt(((o[i]-y[i])**2).mean())) for g,i in groups.items()},flush=True)
best=min(oofs,key=lambda m:np.sqrt(((oofs[m]-y)**2).mean())); print('BEST',best)
# test matrix
tmeta=pd.read_parquet(f"{OUT}/test_meta.parquet"); tmeta['kn']=tmeta.key.map(norm)
tdf=tmeta[['kn','has_pickup','has_dropoff','distance_km']].copy()
for name,vp,tp,_ in MODELS:
    if name in used: tdf=tdf.merge(load_test(tp).rename(columns={'pred':name}),on='kn',how='left')
tdf['lgd']=np.log1p(tdf.distance_km.clip(lower=0).fillna(0)); tdf['year']=tdf.kn.str[:4].astype(float)
tdf['missing']=((tdf.has_pickup==0)|(tdf.has_dropoff==0)); tdf['degen']=(~tdf.missing)&(tdf.distance_km<0.1); tdf['clean']=(~tdf.missing)&(~tdf.degen)
tdf['pred']=np.nan
weights={}
for g,idx in groups.items():
    A=df.loc[idx,used].values; yy=df.loc[idx,'fare_amount'].values
    tm=np.where(tdf[g].values)[0]
    if len(tm)==0: continue
    if best=='linear':
        lr=LinearRegression(positive=True).fit(A,yy); weights[g]={'coef':lr.coef_.tolist(),'intercept':float(lr.intercept_)}
        tdf.loc[tdf.index[tm],'pred']=lr.predict(tdf.iloc[tm][used].values)
    else:
        Xm=df.loc[idx,used+meta_cols].values
        p=dict(objective='regression',metric='rmse',learning_rate=0.05,num_leaves=15,min_data_in_leaf=300,
               feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,num_threads=16,verbosity=-1,seed=1)
        m=lgb.train(p,lgb.Dataset(Xm,yy),num_boost_round=250)
        tdf.loc[tdf.index[tm],'pred']=m.predict(tdf.iloc[tm][used+meta_cols].values)
        weights[g]={'lgb':True,'rounds':250}
if tdf.pred.isna().any():
    tdf.loc[tdf.pred.isna(),'pred']=np.clip(tdf.loc[tdf.pred.isna(),used].mean(axis=1),2.5,200)
tdf['pred']=np.clip(tdf.pred,2.5,200)
test=pd.read_csv(f"{DATA}/test.csv"); test['kn']=test.key.map(norm)
sub=test[['key','kn']].merge(tdf[['kn','pred']],on='kn',how='left').rename(columns={'pred':'fare_amount'})[['key','fare_amount']]
assert sub.fare_amount.notna().all()
sub.to_csv(f"{OUT}/blend_submission.csv",index=False)
os.makedirs('/download_job_code/payload/sandbox/final_solution',exist_ok=True)
sub.to_csv('/download_job_code/payload/sandbox/final_solution/submission.csv',index=False)
json.dump({'models':used,'best':best,'oof':{m:float(np.sqrt(((o-y)**2).mean())) for m,o in oofs.items()},
           'oof_groups':{m:{g:float(np.sqrt(((o[i]-y[i])**2).mean())) for g,i in groups.items()} for m,o in oofs.items()},
           'weights':weights},open(f"{OUT}/blend_weights.json",'w'),indent=2)
print('FINAL submission',sub.shape,sub.fare_amount.describe()[['mean','min','max']].to_dict(),flush=True)
