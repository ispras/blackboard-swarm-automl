import numpy as np, pandas as pd, pyarrow.parquet as pq, glob, os
from scipy.optimize import nnls
W='/download_job_code/payload/sandbox/workspaces'
def canon(s):
    a=s.str.split('.',n=1)
    dt=a.str[0]; fr=a.str[1].fillna('0').str.ljust(9,'0')
    return dt+'.'+fr
vm=pq.read_table(f'{W}/antares_omicron/feat/val_meta.parquet').to_pandas()
vm['ck']=canon(vm['key']); vm=vm.set_index('ck')
miss=(vm['has_pickup'].values<1)|(vm['has_dropoff'].values<1)
y=vm['fare'].values.astype(np.float64)
models={}
files={
 'xgb_te':['antares_omicron/runs/xgb_te_val.parquet','pred'],
 'xgb_lg':['antares_omicron/runs/xgb_lg_val.parquet','pred'],
 'xgb_v2':['antares_omicron/runs/xgb_v2_val.parquet','pred'],
 'xgb_v3':['antares_omicron/runs/xgb_v3_val.parquet','pred'],
 'cat':['antares_omicron/runs/cat_val.parquet','pred'],
 'lgb20m':['arcturus_upsilon/runs/lgb20m_1023_val.parquet','pred'],
 'lgb30m':['arcturus_upsilon/runs/lgb30m_val.parquet','pred'],
 'cat8m_cpu':['arcturus_upsilon/runs/cat8m_cpu_val.parquet','pred'],
 'alkaid_v2lgb':['alkaid_phi/v2lgb_val.parquet','pred'],
}
for name,(f,c) in files.items():
    t=pq.read_table(f'{W}/{f}').to_pandas(); t['ck']=canon(t['key'])
    s=t.drop_duplicates('ck').set_index('ck')[c].rename(name)
    models[name]=s
df=pd.DataFrame(index=vm.index)
for name,s in models.items(): df[name]=s.reindex(df.index)
df['y']=y; df['miss']=miss
print('joined', len(df), 'nonnull per model:\n', df.notna().sum())
def rmse(p,yy): return float(np.sqrt(np.mean((p-yy)**2)))
def fit_eval(model_list, tag, pred_override=None):
    sub=df.dropna(subset=model_list+['y'])
    yy=sub['y'].values; mm=sub['miss'].values
    p=np.full(len(sub),np.nan)
    for grp in [False,True]:
        sel=mm==grp
        if sel.sum()<20: continue
        A=np.column_stack([sub.loc[sel,model_list].values, np.ones(sel.sum())])
        w,_=nnls(A, yy[sel]); p[sel]=A@w
    p=np.clip(p,2.5,200)
    print(tag,'n',len(sub),'all',round(rmse(p,yy),4),'nonmiss',round(rmse(p[~mm],yy[~mm]),4),'miss',round(rmse(p[mm],yy[mm]),4))
    return p
print('\n--- single model ---')
for m in files: 
    sub=df.dropna(subset=[m])
    print(m, round(rmse(np.clip(sub[m].values,2.5,200),sub['y'].values),4), 'n',len(sub))
print('\n--- blends ---')
fit_eval(['xgb_te','xgb_lg','alkaid_v2lgb','lgb30m','lgb20m'],'arcturus_best')
fit_eval(['xgb_te','xgb_lg','alkaid_v2lgb','lgb30m','lgb20m','cat'],'+cat')
fit_eval(['xgb_te','xgb_lg','cat','alkaid_v2lgb','lgb30m','lgb20m','cat8m_cpu'],'+cat8m')
fit_eval(['xgb_te','xgb_lg','cat','xgb_v2','xgb_v3','alkaid_v2lgb','lgb30m','lgb20m','cat8m_cpu'],'all9')
