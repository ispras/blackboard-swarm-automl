import os, sys, json
import numpy as np, pandas as pd
from scipy.optimize import nnls

BASE='/download_job_code/payload/sandbox'
HOL=f'{BASE}/workspaces/arcturus_upsilon/runs/holdout_flags.parquet'
TEST=f'{BASE}/workspaces/arcturus_upsilon/cache/test_v2.parquet'
VAL={
 'lgb20m':'workspaces/arcturus_upsilon/runs/lgb20m_1023_val.parquet',
 'lgb30m':'workspaces/arcturus_upsilon/runs/lgb30m_val.parquet',
 'xgb_v2':'workspaces/antares_omicron/runs/xgb_v2_val.parquet',
 'xgb_te':'workspaces/antares_omicron/runs/xgb_te_val.parquet',
 'xgb_v3':'workspaces/antares_omicron/runs/xgb_v3_val.parquet',
 'xgb_lg':'workspaces/antares_omicron/runs/xgb_lg_val.parquet',
 'cat':'workspaces/antares_omicron/runs/cat_val.parquet',
 'alkaid_v2lgb':'workspaces/alkaid_phi/v2lgb_val.parquet',
}
TES={
 'lgb20m':'workspaces/arcturus_upsilon/runs/lgb20m_1023_test.parquet',
 'lgb30m':'workspaces/arcturus_upsilon/runs/lgb30m_test.parquet',
 'xgb_v2':'workspaces/antares_omicron/runs/xgb_v2_test.parquet',
 'xgb_te':'workspaces/antares_omicron/runs/xgb_te_test.parquet',
 'xgb_v3':'workspaces/antares_omicron/runs/xgb_v3_test.parquet',
 'xgb_lg':'workspaces/antares_omicron/runs/xgb_lg_test.parquet',
 'cat':'workspaces/antares_omicron/runs/cat_test.parquet',
 'alkaid_v2lgb':'workspaces/alkaid_phi/v2lgb_submission.csv',
}
DEFAULT='xgb_te,xgb_v2,xgb_v3,alkaid_v2lgb,lgb30m,lgb20m'

def nk(ks):
    out=np.empty(len(ks),dtype=object)
    for i,k in enumerate(ks):
        k=str(k); b,f=k.rsplit('.',1) if '.' in k else (k,''); out[i]=b+'.'+f.ljust(9,'0')[:9]
    return out

def load(path):
    df=pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
    pc=next(c for c in ('pred','fare_amount','y_pred') if c in df.columns)
    df=df[['key',pc]].rename(columns={pc:'pred'}); df['key']=nk(df.key.values)
    return df.drop_duplicates('key')

def fitw(X,y):
    A=np.column_stack([X,np.ones(len(y))]); w,_=nnls(A,y); return w[:-1],float(w[-1])

def main(models_arg, out):
    models=[m.strip() for m in models_arg.split(',') if m.strip() in VAL and os.path.exists(os.path.join(BASE,VAL[m.strip()]))]
    hol=pd.read_parquet(HOL); hol['key']=nk(hol.key.values); hol=hol.rename(columns={'fare_amount':'y'}).drop_duplicates('key')
    df=hol[['key','y','has_pickup','has_dropoff','distance_km']].copy()
    for m in models:
        df=df.merge(load(os.path.join(BASE,VAL[m])).rename(columns={'pred':m}),on='key',how='left')
    miss=((df.has_pickup==0)|(df.has_dropoff==0)).to_numpy()
    fnm=~miss & np.isfinite(df[models].to_numpy(float)).all(1)
    wnm,bnm=fitw(df.loc[fnm,models].to_numpy(float),df.loc[fnm,'y'].to_numpy(float))
    print('nonmiss weights:',{m:round(float(w),4) for m,w in zip(models,wnm)},'b',round(bnm,4),'n',int(fnm.sum()))

    # ---- test load ----
    tf=pd.read_parquet(TEST)[['key','has_pickup','has_dropoff','distance_km']].copy()
    orig_key=tf['key'].to_numpy().copy()
    tf['kn']=nk(orig_key); tb=tf.set_index('kn')
    for m in models:
        tb=tb.join(load(os.path.join(BASE,TES[m])).set_index('key').rename(columns={'pred':m}),how='left')
    tmiss=((tb.has_pickup==0)|(tb.has_dropoff==0)).to_numpy()
    # auto safety for missing test rows
    safe=[]
    for m in models:
        col=tb[m].to_numpy(float)[tmiss]
        mm=float(np.nanmean(col)) if np.isfinite(col).any() else 999.0
        if mm<=20.0:
            safe.append(m)
        else:
            print('EXCLUDE from missing blend: %s (test miss mean %.1f)'%(m,mm))
    print('missing blend models:',safe)
    fm=miss & np.isfinite(df[safe].to_numpy(float)).all(1)
    wm,bm=fitw(df.loc[fm,safe].to_numpy(float),df.loc[fm,'y'].to_numpy(float))
    print('missing weights:',{m:round(float(w),4) for m,w in zip(safe,wm)},'b',round(bm,4),'n',int(fm.sum()))

    # ---- report val ----
    pred=np.full(len(df),np.nan)
    pred[fnm]=df.loc[fnm,models].to_numpy(float)@wnm+bnm
    pred[fm]=df.loc[fm,safe].to_numpy(float)@wm+bm
    pred=np.clip(pred,2.5,200.0)
    ok=np.isfinite(pred); y=df.y.to_numpy(); e=pred-y
    def r(m): return float(np.sqrt(np.mean(e[m]**2)))
    print(f'VAL covered={int(ok.sum())}/{len(df)} all={r(ok):.4f} nonmiss={r(ok&~miss):.4f} miss={r(ok&miss):.4f}')

    # ---- apply test ----
    pt=np.full(len(tb),np.nan)
    Xn=tb[models].to_numpy(float); fn=np.isfinite(Xn).all(1); mn=~tmiss&fn
    pt[mn]=Xn[mn]@wnm+bnm
    Xs=tb[safe].to_numpy(float); fs=np.isfinite(Xs).all(1); ms=tmiss&fs
    pt[ms]=Xs[ms]@wm+bm
    rest=~np.isfinite(pt)
    if rest.any():
        pt[rest]=np.nanmean(tb[safe].to_numpy(float)[rest],axis=1)
        print('fallback rows',int(rest.sum()))
    pt=np.clip(pt,2.5,200.0)
    sub=pd.DataFrame({'key':orig_key,'fare_amount':pt}); sub.to_csv(out,index=False)
    print('wrote',out,'n',len(sub),'stats',float(pt.min()),float(np.median(pt)),float(pt.max()),float(pt.mean()))
    print('TEST miss mean',float(pt[tmiss].mean()),'nonmiss mean',float(pt[~tmiss].mean()))
    json.dump({'models':models,'missing_models':safe,
               'w_nonmiss':dict(zip(models,map(float,wnm))),'b_nonmiss':bnm,
               'w_miss':dict(zip(safe,map(float,wm))),'b_miss':bm,
               'val_all':r(ok),'val_nonmiss':r(ok&~miss),'val_miss':r(ok&miss)},
              open(f'{BASE}/workspaces/arcturus_upsilon/runs/final_blend_meta.json','w'),indent=1)

if __name__=='__main__':
    models=sys.argv[1] if len(sys.argv)>1 else DEFAULT
    out=sys.argv[2] if len(sys.argv)>2 else f'{BASE}/workspaces/arcturus_upsilon/runs/final_safe_submission.csv'
    main(models,out)
